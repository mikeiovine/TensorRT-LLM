# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Sweep speculative decoding on DeepSeek-V4-Flash and compare the arms on a Pareto front.

Arms, all on one checkpoint, each swept over global batch sizes and draft lengths K:

  baseline  no speculative decoding
  fixed     --decoding-type (MTP or DSpark) verifying K draft tokens every step
  av        the same drafter with adaptive verification (enable_confidence_scheduling),
            maximum K

The checkpoint must hold the drafter of --decoding-type: MTP layers for MTP (the default), a
DSpark drafter for DSpark.

Stages, run in this order by "all":

  data     build the ShareGPT dataset with build_datasets.py
  costs    profile the cost table adaptive verification needs (dspark_profile_costs)
  run      one trtllm-bench run per (arm, K, batch size); finished runs are skipped
  report   table, CSV and Pareto plot of tok/s/user against tok/s/GPU

Every run uses TP = EP = --gpus with attention DP, so the rank-local batch size is the global batch
size / --gpus. Requests are sent at constant concurrency equal to the batch size, greedy, and stop
at EOS or after 1,024 output tokens.

Example on 4x B300:

  python sweep_spec_decode.py all --model /models/DeepSeek-V4-Flash --work-dir sweep
"""

import argparse
import csv
import json
import os
import subprocess  # nosec B404 - runs fixed argv lists, no shell
import sys
from collections import defaultdict
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from analyze import frontier, iso_gain  # noqa: E402

ARMS = ["baseline", "fixed", "av"]
GRAPH_SIZES = (4, 8, 16, 32, 64, 128, 256)
# Pairs compared at equal per-user speed: (arm, reference).
COMPARISONS = [("fixed", "baseline"), ("av", "fixed"), ("av", "baseline")]


def graph_ladder(local: int) -> list[int]:
    return [g for g in GRAPH_SIZES if g < local] + [local]


def base_config(args, local: int) -> dict:
    config = {
        "gpus_per_node": args.gpus,
        "tensor_parallel_size": args.gpus,
        "moe_expert_parallel_size": args.gpus,
        "moe_tensor_parallel_size": 1,
        "enable_attention_dp": True,
        "attention_dp_config": {
            "enable_balance": True,
            "batching_wait_iters": 10,
            "timeout_iters": 50,
        },
        "max_batch_size": local,
        "max_num_tokens": 4096,
        "max_input_len": 1536,
        "max_seq_len": 2560,
        "enable_chunked_prefill": False,
        "custom_tokenizer": "deepseek_v4",
        "kv_cache_config": {
            "dtype": "fp8",
            "free_gpu_memory_fraction": args.kv_fraction,
            "enable_block_reuse": False,
            "tokens_per_block": 128,
            "host_cache_size": 0,
        },
        "moe_config": {"backend": args.moe_backend},
        "cuda_graph_config": {
            "enable_padding": True,
            "max_batch_size": local,
            "batch_sizes": graph_ladder(local),
        },
        "prefill_cuda_graph_backend": "breakable",
    }
    if args.extra_config:
        config.update(yaml.safe_load(args.extra_config.read_text()) or {})
    return config


def spec_config(args, arm: str, k: int) -> dict | None:
    if arm == "baseline":
        return None
    spec = {"decoding_type": args.decoding_type, "max_draft_len": k}
    if args.decoding_type == "DSpark":
        spec["block_size"] = k
    if arm == "av":
        spec.update(
            enable_confidence_scheduling=True,
            confidence_cost_table_path=str(cost_table(args)),
            enable_fused_confidence_scheduler=not args.host_ranking,
        )
        if args.calibration:
            spec["confidence_calibration_path"] = str(args.calibration.resolve())
    return spec


def cost_table(args) -> Path:
    return (args.cost_table or args.work_dir / "costs.json").resolve()


def dataset_path(args) -> Path:
    return (args.dataset or args.work_dir / "data" / "sharegpt.jsonl").resolve()


def plan(args) -> list[tuple[str, int, int, int]]:
    """(arm, K, batch size, repeat) of every run, grouped so one arm's runs are adjacent."""
    draft_lens = {"fixed": args.fixed_k, "av": args.av_k}
    runs = []
    for arm in args.arms:
        for k in draft_lens.get(arm, [0]):
            for batch_size in args.batch_sizes:
                for repeat in range(1, args.repeats + 1):
                    runs.append((arm, k, batch_size, repeat))
    return runs


def run_dir(args, arm: str, k: int, batch_size: int, repeat: int) -> Path:
    name = f"{arm}-k{k}-bs{batch_size}-r{repeat}"
    return args.work_dir / "runs" / dataset_path(args).stem / name


def check_batch_sizes(args) -> None:
    bad = [bs for bs in args.batch_sizes if bs % args.gpus or bs < args.gpus]
    if bad:
        sys.exit(f"batch sizes must be multiples of --gpus {args.gpus}: {bad}")


def launch(command: list[str], dry_run: bool, log: Path | None = None, env=None) -> int:
    print("+ " + " ".join(command), flush=True)
    if dry_run:
        return 0
    if log is None:
        return subprocess.run(command, env=env, check=False).returncode  # nosec B603
    with log.open("w") as file:
        return subprocess.run(  # nosec B603
            command, env=env, stdout=file, stderr=subprocess.STDOUT, check=False
        ).returncode


def stage_data(args) -> None:
    path = dataset_path(args)
    if path.exists():
        print(f"dataset: {path} exists")
        return
    if args.dataset:
        sys.exit(f"{path} does not exist")
    command = [sys.executable, str(HERE / "build_datasets.py"), "--model", str(args.model)]
    command += ["--output-dir", str(path.parent), "--datasets", "sharegpt"]
    code = launch(command, args.dry_run)
    # build_datasets.py exits nonzero when the rows differ from the published
    # Pro dataset, which another checkpoint's tokenizer can cause.
    if code and not path.exists():
        sys.exit(f"build_datasets.py failed with exit code {code}")


def stage_costs(args) -> None:
    if "av" not in args.arms:
        return
    locals_ = [bs // args.gpus for bs in args.batch_sizes]
    ladder = sorted({g for local in locals_ for g in graph_ladder(local)})
    path = cost_table(args)
    costs = json.loads(path.read_text())["costs"] if path.exists() else {}
    missing = [k for k in args.av_k if any(str(g) not in costs.get(str(k), {}) for g in ladder)]
    if not missing:
        print(f"cost table: {path} covers K {args.av_k} at G {ladder}")
        return
    if args.cost_table:
        sys.exit(f"{path} has no rows for K {missing} at every G of {ladder}")
    config = base_config(args, max(locals_))
    config["speculative_config"] = {
        "decoding_type": args.decoding_type,
        "max_draft_len": max(missing),
    }
    config_path = args.work_dir / "costs-config.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    command = [sys.executable, "-m", "tensorrt_llm.tools.dspark_profile_costs"]
    command += ["--model", str(args.model), "--config", str(config_path)]
    command += ["--output", str(path), "--dataset", str(dataset_path(args))]
    command += ["--k", *map(str, missing), "--graph-batch-sizes", *map(str, ladder)]
    if launch(command, args.dry_run):
        sys.exit("cost profiling failed")


def stage_run(args) -> None:
    runs = plan(args)
    failed = []
    for index, (arm, k, batch_size, repeat) in enumerate(runs, 1):
        out = run_dir(args, arm, k, batch_size, repeat)
        if (out / "report.json").exists():
            continue
        local = batch_size // args.gpus
        config = base_config(args, local)
        spec = spec_config(args, arm, k)
        if spec is not None:
            config["speculative_config"] = spec
        model = str(args.model)
        num_requests = min(max(batch_size * 20, 320), 10240)
        warmup = min(max(batch_size * 2, 64), 1024)
        command = ["trtllm-bench", "--model", model, "--model_path", model]
        command += ["--workspace", str(out / "workspace"), "throughput", "--backend", "pytorch"]
        command += ["--config", str(out / "config.yaml")]
        command += ["--sampler_options", str(HERE / "sampler.yaml")]
        command += ["--dataset", str(dataset_path(args)), "--num_requests", str(num_requests)]
        command += ["--concurrency", str(batch_size), "--warmup", str(warmup)]
        command += ["--tp", str(args.gpus), "--pp", "1", "--ep", str(args.gpus)]
        command += ["--max_batch_size", str(local)]
        command += ["--max_num_tokens", str(config["max_num_tokens"])]
        command += ["--max_seq_len", str(config["max_seq_len"])]
        command += ["--kv_cache_free_gpu_mem_fraction", str(args.kv_fraction)]
        command += ["--disable_chunked_context", "--custom_tokenizer", "deepseek_v4"]
        command += ["--eos_id", "1", "--beam_width", "1"]
        command += ["--report_json", str(out / "report.json")]
        command += ["--request_json", str(out / "requests.json")]
        print(f"[{index}/{len(runs)}] {out.name}", flush=True)
        if not args.dry_run:
            out.mkdir(parents=True, exist_ok=True)
            (out / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
            meta = {"arm": arm, "k": k, "batch_size": batch_size, "gpus": args.gpus}
            (out / "run.json").write_text(json.dumps(meta))
        # The allocator setting of the published DSpark results.
        env = dict(os.environ, PYTORCH_ALLOC_CONF="garbage_collection_threshold:0.99999")
        code = launch(command, args.dry_run, out / "log.txt", env)
        if code or not (args.dry_run or (out / "report.json").exists()):
            failed.append(out)
            print(f"  FAILED (exit code {code}), see {out / 'log.txt'}", flush=True)
            continue
        if not args.dry_run:
            print("  " + describe(load_run(out)), flush=True)
    if failed:
        print(f"{len(failed)} of {len(runs)} runs failed; rerun to retry them")


def load_run(out: Path) -> dict:
    meta = json.loads((out / "run.json").read_text())
    perf = json.loads((out / "report.json").read_text())["performance"]
    return {
        "arm": meta["arm"],
        "k": meta["k"],
        "bs": meta["batch_size"],
        "tps_gpu": perf["system_output_throughput_tok_s"] / meta["gpus"],
        "tps_user": perf["output_throughput_per_user_tok_s"],
    }


def describe(point: dict) -> str:
    return f"{point['tps_gpu']:.0f} tok/s/GPU, {point['tps_user']:.1f} tok/s/user"


def label(point: dict) -> str:
    return f"bs{point['bs']}" + (f" K{point['k']}" if point["k"] else "")


def stage_report(args) -> None:
    root = args.work_dir / "runs" / dataset_path(args).stem
    groups = defaultdict(list)
    for report in sorted(root.glob("*/report.json")):
        run = load_run(report.parent)
        groups[(run["arm"], run["k"], run["bs"])].append(run)
    if not groups:
        sys.exit(f"no finished runs under {root}")
    points = [
        {
            "arm": arm,
            "k": k,
            "bs": bs,
            "n": len(runs),
            "tps_gpu": sum(r["tps_gpu"] for r in runs) / len(runs),
            "tps_user": sum(r["tps_user"] for r in runs) / len(runs),
        }
        for (arm, k, bs), runs in sorted(groups.items())
    ]
    csv_path = args.work_dir / f"results-{root.name}.csv"
    with csv_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(points[0]))
        writer.writeheader()
        writer.writerows(points)

    arms = [arm for arm in ARMS if any(p["arm"] == arm for p in points)]
    fronts = {arm: frontier([p for p in points if p["arm"] == arm]) for arm in arms}
    for arm in arms:
        print(f"\n== {arm}: Pareto front (of {sum(p['arm'] == arm for p in points)} points)")
        for point in fronts[arm]:
            print(f"  {label(point):>12}  {describe(point)}  (n={point['n']})")
    print("\n== tok/s/GPU at equal tok/s/user, between the two fronts")
    for arm, reference in COMPARISONS:
        curve = iso_gain(fronts.get(reference, []), fronts.get(arm, []))
        if not curve:
            continue
        gains = [gain for *_, gain in curve]
        x, ref, new, gain = max(curve, key=lambda row: row[3])
        print(
            f"  {arm} vs {reference}: {min(gains):+.1f}% to {max(gains):+.1f}% over "
            f"{curve[0][0]:.1f}-{curve[-1][0]:.1f} tok/s/user; largest at {x:.1f} "
            f"({ref:.0f} -> {new:.0f} tok/s/GPU)"
        )
    print(f"\nwrote {csv_path}")
    if args.plot:
        plot(points, fronts, args.plot, root.name)
        print(f"wrote {args.plot}")


def plot(points: list[dict], fronts: dict, path: Path, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"baseline": "tab:gray", "fixed": "tab:green", "av": "tab:blue"}
    fig, ax = plt.subplots(figsize=(9, 6))
    for arm, line in fronts.items():
        scattered = [p for p in points if p["arm"] == arm]
        ax.scatter(
            [p["tps_user"] for p in scattered],
            [p["tps_gpu"] for p in scattered],
            s=14,
            color=colors[arm],
            alpha=0.35,
        )
        ax.plot(
            [p["tps_user"] for p in line],
            [p["tps_gpu"] for p in line],
            color=colors[arm],
            marker="o",
            ms=4,
            label=arm,
        )
        for point in line:
            ax.annotate(
                label(point),
                (point["tps_user"], point["tps_gpu"]),
                fontsize=7,
                textcoords="offset points",
                xytext=(0, 6),
                ha="center",
                color=colors[arm],
            )
    ax.set_title(title)
    ax.set_xlabel("output tok/s per user")
    ax.set_ylabel("output tok/s per GPU")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("stage", choices=["data", "costs", "run", "report", "all"])
    parser.add_argument("--model", type=Path, help="checkpoint holding the drafter")
    parser.add_argument("--decoding-type", choices=["MTP", "DSpark"], default="MTP")
    parser.add_argument("--work-dir", type=Path, default=Path("sweep"))
    parser.add_argument("--gpus", type=int, default=4)
    parser.add_argument("--arms", nargs="+", choices=ARMS, default=ARMS)
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=[16, 32, 64, 128, 256, 512, 1024],
        help="global batch sizes, multiples of --gpus",
    )
    parser.add_argument("--fixed-k", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--av-k", type=int, nargs="+", default=[2, 3, 4, 5])
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--dataset", type=Path, help="trtllm-bench JSONL (default: ShareGPT)")
    parser.add_argument("--cost-table", type=Path, help="existing cost table for this system")
    parser.add_argument("--calibration", type=Path, help="STS file of a trained confidence head")
    parser.add_argument(
        "--host-ranking", action="store_true", help="av arm without device re-ranking"
    )
    parser.add_argument("--moe-backend", default="TRTLLM")
    parser.add_argument("--kv-fraction", type=float, default=0.5)
    parser.add_argument("--extra-config", type=Path, help="YAML merged over every runtime config")
    parser.add_argument("--plot", type=Path, help="PNG of the Pareto fronts (report stage)")
    parser.add_argument("--dry-run", action="store_true", help="print the commands only")
    args = parser.parse_args()

    check_batch_sizes(args)
    if args.calibration and args.decoding_type != "DSpark":
        sys.exit("--calibration applies to a DSpark drafter with a trained confidence head")
    if args.av_k and min(args.av_k) < 2:
        sys.exit("--av-k values must be at least 2: K1 has no shorter verification window")
    args.work_dir = args.work_dir.resolve()
    if args.stage != "report":
        if args.model is None or not args.model.is_dir():
            sys.exit("--model must be a checkpoint directory")
        args.model = args.model.resolve()
    if args.plot is None and args.stage in ("report", "all"):
        args.plot = args.work_dir / "pareto.png"
    stages = ["data", "costs", "run", "report"] if args.stage == "all" else [args.stage]
    for stage in stages:
        if stage == "report" and args.dry_run:
            continue
        {"data": stage_data, "costs": stage_costs, "run": stage_run, "report": stage_report}[
            stage
        ](args)


if __name__ == "__main__":
    main()
