# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PEARL: Parallel Speculative Decoding with Adaptive Draft Length — PoC

Implements Algorithm 2 from https://arxiv.org/pdf/2408.11850 (ICLR 2025).

Layout
------
* Target model  — spread across N GPUs via HuggingFace ``device_map``
                  (pipeline-parallel style, layers partitioned across devices).
* Draft model   — loaded on a separate GPU so it can run concurrently.

Parallelism is achieved with a two-thread pool: one thread drives the draft
model, the other drives the target model.  Because they live on disjoint GPUs,
CUDA kernels execute truly in parallel (Python threads release the GIL during
CUDA calls).

Two modes alternate during generation:

* **pre-verify** — draft produces γ tokens while the target evaluates the
  current prefix.  The target's logits are used to accept/reject the *first*
  draft token.  If accepted the draft batch becomes "pending" and we switch to
  post-verify.  If rejected we resample one token and stay in pre-verify.

* **post-verify** — the target verifies the pending γ tokens while the draft
  continues to produce γ *more* tokens speculatively.  If all pending tokens
  are accepted the new draft batch becomes the next pending batch (staying in
  post-verify).  Otherwise we keep the accepted prefix, resample one token,
  and switch back to pre-verify.

Usage
-----
    python pearl_poc.py \
        --target-model meta-llama/Llama-3.1-70B-Instruct \
        --draft-model  meta-llama/Llama-3.2-1B-Instruct  \
        --target-gpus 0 1 2 3 \
        --draft-gpu 4 \
        --gamma 5 \
        --max-new-tokens 200 \
        --prompt "The future of artificial intelligence is" \
        --compare-ar
"""

from __future__ import annotations

import argparse
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---------------------------------------------------------------------------
# Statistics tracker
# ---------------------------------------------------------------------------


@dataclass
class PEARLStats:
    pre_verify_steps: int = 0
    post_verify_steps: int = 0
    accepted_tokens: int = 0
    rejected_first_tokens: int = 0
    total_generated: int = 0
    draft_forwards: int = 0
    target_forwards: int = 0
    wall_time_s: float = 0.0

    def summary(self) -> str:
        total_steps = self.pre_verify_steps + self.post_verify_steps
        tps = (self.total_generated / self.wall_time_s
               if self.wall_time_s > 0 else 0.0)
        accept_rate = (
            self.accepted_tokens /
            max(self.accepted_tokens + self.rejected_first_tokens, 1))
        return "\n".join([
            "─── PEARL Statistics ───",
            f"  Tokens generated      : {self.total_generated}",
            f"  Wall time             : {self.wall_time_s:.3f} s",
            f"  Throughput            : {tps:.1f} tok/s",
            f"  Pre-verify steps      : {self.pre_verify_steps}",
            f"  Post-verify steps     : {self.post_verify_steps}",
            f"  Total decoding steps  : {total_steps}",
            f"  Mean tokens / step    : "
            f"{self.total_generated / max(total_steps, 1):.2f}",
            f"  Accepted tokens       : {self.accepted_tokens}",
            f"  Rejected (first tok)  : {self.rejected_first_tokens}",
            f"  First-token accept %  : {accept_rate:.1%}",
            f"  Draft model forwards  : {self.draft_forwards}",
            f"  Target model forwards : {self.target_forwards}",
        ])


# ---------------------------------------------------------------------------
# PEARL decoder
# ---------------------------------------------------------------------------


class PEARLDecoder:
    """Disaggregated PEARL speculative decoder.

    The target and draft models live on separate GPUs so their forward passes
    overlap via a thread pool.
    """

    def __init__(
        self,
        target_model_name: str,
        draft_model_name: str,
        target_gpus: list[int],
        draft_gpu: int,
        gamma: int = 5,
        dtype: torch.dtype = torch.bfloat16,
    ):
        self.gamma = gamma
        self.draft_device = torch.device(f"cuda:{draft_gpu}")
        self._pool = ThreadPoolExecutor(max_workers=2)

        print(f"[PEARL] Loading draft model '{draft_model_name}' → GPU {draft_gpu}")
        self.tokenizer = AutoTokenizer.from_pretrained(target_model_name)
        self.draft_model = AutoModelForCausalLM.from_pretrained(
            draft_model_name,
            torch_dtype=dtype,
        ).to(self.draft_device).eval()

        print(
            f"[PEARL] Loading target model '{target_model_name}' → "
            f"GPUs {target_gpus}"
        )
        max_memory = {g: "auto" for g in target_gpus}
        self.target_model = AutoModelForCausalLM.from_pretrained(
            target_model_name,
            torch_dtype=dtype,
            device_map="balanced",
            max_memory=max_memory,
        ).eval()

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    # ── Draft helpers ─────────────────────────────────────────────────────

    def _draft_n_tokens(
        self,
        prefix_ids: torch.Tensor,
        n: int,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Auto-regressively sample *n* tokens from the draft model.

        Returns
        -------
        tokens : Tensor, shape ``(n,)``
        probs  : list of ``n`` Tensors, each shape ``(vocab,)``
        """
        ids = prefix_ids.to(self.draft_device)
        tokens: list[torch.Tensor] = []
        probs: list[torch.Tensor] = []
        past = None

        for _ in range(n):
            out = self.draft_model(
                ids if past is None else ids[:, -1:],
                past_key_values=past,
                use_cache=True,
            )
            past = out.past_key_values
            logits = out.logits[:, -1, :]
            p = torch.softmax(logits, dim=-1).squeeze(0)
            tok = torch.multinomial(p.unsqueeze(0), num_samples=1).squeeze()
            tokens.append(tok)
            probs.append(p)
            ids = torch.cat([ids, tok.view(1, 1)], dim=-1)

        return torch.stack(tokens), probs

    # ── Target helpers ────────────────────────────────────────────────────

    def _target_forward(self, ids: torch.Tensor) -> torch.Tensor:
        """Single target-model forward.  Returns logits ``(1, L, V)``."""
        first_param = next(self.target_model.parameters())
        return self.target_model(ids.to(first_param.device)).logits

    # ── Speculative sampling ──────────────────────────────────────────────

    @staticmethod
    def _spec_accept(
        p: torch.Tensor,
        q: torch.Tensor,
        tok: torch.Tensor,
    ) -> tuple[bool, Optional[torch.Tensor]]:
        """Modified rejection sampling (Leviathan et al., 2023).

        Returns ``(accepted, replacement_token | None)``.
        """
        p_t = p[tok]
        q_t = q[tok]

        if q_t > 0:
            ratio = (p_t / q_t).clamp(max=1.0)
        else:
            ratio = torch.tensor(1.0, device=p.device) if p_t > 0 else torch.tensor(0.0, device=p.device)

        if torch.rand(1, device=p.device) <= ratio:
            return True, None

        residual = (p - q).clamp(min=0)
        total = residual.sum()
        if total > 0:
            residual = residual / total
        else:
            residual = p
        replacement = torch.multinomial(residual.unsqueeze(0), 1).squeeze()
        return False, replacement

    # ── Main PEARL generation loop ────────────────────────────────────────

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 200,
    ) -> tuple[str, PEARLStats]:
        """Generate text using Algorithm 2 of the PEARL paper."""

        input_ids = self.tokenizer.encode(prompt, return_tensors="pt")
        prompt_len = input_ids.shape[1]

        # `confirmed` holds only *verified* token ids.
        confirmed: list[int] = input_ids.squeeze(0).tolist()

        mode = "pre-verify"
        stats = PEARLStats()

        # Draft tokens awaiting verification (set when entering post-verify).
        pending_tokens: list[int] = []
        pending_probs: list[torch.Tensor] = []

        eos_id = self.tokenizer.eos_token_id
        t0 = time.perf_counter()

        while (len(confirmed) - prompt_len) < max_new_tokens:
            if mode == "pre-verify":
                stats.pre_verify_steps += 1
                prefix = torch.tensor([confirmed], dtype=torch.long)

                # ── parallel: draft γ tokens | target forward on prefix ──
                fut_draft = self._pool.submit(
                    self._draft_n_tokens, prefix, self.gamma)
                fut_target = self._pool.submit(self._target_forward, prefix)

                draft_tokens, draft_probs = fut_draft.result()
                target_logits = fut_target.result()
                stats.draft_forwards += self.gamma
                stats.target_forwards += 1

                # Verify x_1 using p = softmax(target_logits[last_prefix_pos])
                p = torch.softmax(
                    target_logits[0, -1, :], dim=-1,
                ).to(self.draft_device)
                q = draft_probs[0]
                x1 = draft_tokens[0]

                accepted, replacement = self._spec_accept(p, q, x1)

                if accepted:
                    # First token good — tentatively treat all γ as pending.
                    pending_tokens = draft_tokens.tolist()
                    pending_probs = draft_probs
                    mode = "post-verify"
                else:
                    confirmed.append(replacement.item())
                    stats.rejected_first_tokens += 1

            else:  # ── post-verify ───────────────────────────────────────
                stats.post_verify_steps += 1

                # Build full prefix = confirmed + pending draft tokens
                full_ids = confirmed + pending_tokens
                full_prefix = torch.tensor([full_ids], dtype=torch.long)

                # ── parallel: target verifies | draft continues ──────────
                fut_target = self._pool.submit(
                    self._target_forward, full_prefix)
                fut_draft = self._pool.submit(
                    self._draft_n_tokens, full_prefix, self.gamma)

                target_logits = fut_target.result()
                new_draft_tokens, new_draft_probs = fut_draft.result()
                stats.target_forwards += 1
                stats.draft_forwards += self.gamma

                # Verify each pending token sequentially (standard spec
                # sampling).  Logit at position (base-1+i) predicts position
                # (base+i) where base = len(confirmed).
                base = len(confirmed)
                n_accepted = 0
                rejection_replacement: Optional[torch.Tensor] = None

                for i in range(len(pending_tokens)):
                    logit_pos = base - 1 + i
                    if logit_pos < 0 or logit_pos >= target_logits.shape[1]:
                        break

                    p = torch.softmax(
                        target_logits[0, logit_pos, :], dim=-1,
                    ).to(self.draft_device)
                    q = pending_probs[i]
                    xi = torch.tensor(
                        pending_tokens[i], device=self.draft_device)

                    accepted, replacement = self._spec_accept(p, q, xi)
                    if accepted:
                        n_accepted += 1
                    else:
                        rejection_replacement = replacement
                        break

                if n_accepted == len(pending_tokens):
                    # All γ tokens accepted → confirm them, queue new drafts.
                    confirmed.extend(pending_tokens)
                    stats.accepted_tokens += n_accepted

                    # The target also produced a "bonus" logit at the last
                    # position which predicts the token after the last pending
                    # token.  Use it to pre-verify the first new draft token
                    # (the PEARL "embedded pre-verify").
                    bonus_pos = len(full_ids) - 1
                    if bonus_pos < target_logits.shape[1]:
                        p_bonus = torch.softmax(
                            target_logits[0, bonus_pos, :], dim=-1,
                        ).to(self.draft_device)
                        x_next = torch.tensor(
                            new_draft_tokens[0].item(),
                            device=self.draft_device,
                        )
                        q_next = new_draft_probs[0]
                        acc_bonus, rep_bonus = self._spec_accept(
                            p_bonus, q_next, x_next)

                        if acc_bonus:
                            # First new draft token also accepted — stay in
                            # post-verify with the new batch as pending.
                            pending_tokens = new_draft_tokens.tolist()
                            pending_probs = new_draft_probs
                            mode = "post-verify"
                        else:
                            confirmed.append(rep_bonus.item())
                            stats.rejected_first_tokens += 1
                            pending_tokens = []
                            pending_probs = []
                            mode = "pre-verify"
                    else:
                        pending_tokens = new_draft_tokens.tolist()
                        pending_probs = new_draft_probs
                        mode = "post-verify"
                else:
                    # Partial acceptance — confirm accepted prefix + resample.
                    confirmed.extend(pending_tokens[:n_accepted])
                    stats.accepted_tokens += n_accepted
                    if rejection_replacement is not None:
                        confirmed.append(rejection_replacement.item())
                    stats.rejected_first_tokens += 1
                    pending_tokens = []
                    pending_probs = []
                    mode = "pre-verify"

            # Early stop on EOS
            if eos_id is not None and confirmed[-1] == eos_id:
                break

        stats.wall_time_s = time.perf_counter() - t0
        stats.total_generated = len(confirmed) - prompt_len
        text = self.tokenizer.decode(confirmed, skip_special_tokens=True)
        return text, stats

    # ── Autoregressive baseline (target only) ─────────────────────────────

    @torch.no_grad()
    def generate_autoregressive(
        self,
        prompt: str,
        max_new_tokens: int = 200,
    ) -> tuple[str, float, int]:
        """Vanilla autoregressive generation with the target model."""
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt")
        prompt_len = input_ids.shape[1]
        dev = next(self.target_model.parameters()).device
        ids = input_ids.to(dev)

        t0 = time.perf_counter()
        past = None
        for _ in range(max_new_tokens):
            out = self.target_model(
                ids if past is None else ids[:, -1:],
                past_key_values=past,
                use_cache=True,
            )
            past = out.past_key_values
            tok = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            ids = torch.cat([ids, tok], dim=-1)
            if tok.item() == self.tokenizer.eos_token_id:
                break

        wall = time.perf_counter() - t0
        n_gen = ids.shape[1] - prompt_len
        text = self.tokenizer.decode(ids.squeeze(), skip_special_tokens=True)
        return text, wall, n_gen


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="PEARL Speculative Decoding — Proof of Concept",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--target-model", required=True,
        help="HF model id or local path for the target (large) model.")
    parser.add_argument(
        "--draft-model", required=True,
        help="HF model id or local path for the draft (small) model.")
    parser.add_argument(
        "--target-gpus", type=int, nargs="+", default=[0, 1, 2, 3],
        help="GPU indices for the target model (default: 0 1 2 3).")
    parser.add_argument(
        "--draft-gpu", type=int, default=4,
        help="GPU index for the draft model (default: 4).")
    parser.add_argument(
        "--gamma", type=int, default=5,
        help="Window size γ — number of draft tokens per step (default: 5).")
    parser.add_argument(
        "--max-new-tokens", type=int, default=200,
        help="Maximum tokens to generate (default: 200).")
    parser.add_argument(
        "--prompt", type=str,
        default="The future of artificial intelligence is",
        help="Input prompt.")
    parser.add_argument(
        "--compare-ar", action="store_true",
        help="Also run autoregressive baseline and report speedup.")
    args = parser.parse_args()

    decoder = PEARLDecoder(
        target_model_name=args.target_model,
        draft_model_name=args.draft_model,
        target_gpus=args.target_gpus,
        draft_gpu=args.draft_gpu,
        gamma=args.gamma,
    )

    print(f"\n[PEARL] γ = {args.gamma}, max_new_tokens = {args.max_new_tokens}")
    print(f"[PEARL] Prompt: {args.prompt!r}\n")

    # ── PEARL generation ──────────────────────────────────────────────────
    text, stats = decoder.generate(args.prompt,
                                   max_new_tokens=args.max_new_tokens)
    print("── PEARL output ──")
    print(text)
    print()
    print(stats.summary())

    # ── Optional autoregressive comparison ────────────────────────────────
    if args.compare_ar:
        print("\n── Autoregressive baseline ──")
        ar_text, ar_wall, ar_n = decoder.generate_autoregressive(
            args.prompt, max_new_tokens=args.max_new_tokens)
        ar_tps = ar_n / ar_wall if ar_wall > 0 else 0
        print(f"  Tokens: {ar_n}  Time: {ar_wall:.3f}s  "
              f"Throughput: {ar_tps:.1f} tok/s")
        print(f"  Output: {ar_text[:200]}{'…' if len(ar_text) > 200 else ''}")

        if ar_tps > 0 and stats.wall_time_s > 0:
            pearl_tps = stats.total_generated / stats.wall_time_s
            print(f"\n  ➜ PEARL speedup: {pearl_tps / ar_tps:.2f}×")


if __name__ == "__main__":
    main()
