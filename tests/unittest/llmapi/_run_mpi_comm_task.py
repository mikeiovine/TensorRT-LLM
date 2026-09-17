# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import os
import time
from pathlib import Path
from typing import Literal

import click
from _flashinfer_workspace_probe import get_flashinfer_environment
from _rank_failure_probe import (RANK_1_FAILURE_MESSAGE,
                                 fail_on_rank_1_while_rank_0_runs)

from tensorrt_llm.executor.utils import get_spawn_proxy_process_ipc_hmac_key_env
from tensorrt_llm.llmapi.mpi_session import (MpiPoolSession,
                                             RemoteMpiCommSessionClient)
from tensorrt_llm.llmapi.utils import print_colored


@click.command()
@click.option("--task_type",
              type=click.Choice([
                  "submit", "submit_sync", "flashinfer_workspace",
                  "flashinfer_temporary_cleanup", "async_rank_failure"
              ]),
              default="submit")
def main(
    task_type: Literal["submit", "submit_sync", "flashinfer_workspace",
                       "flashinfer_temporary_cleanup", "async_rank_failure"]
) -> None:
    """Run the requested remote MPI session test task."""
    # TODO(dlfw-26.08): drop once the nested-spawn failure is settled. The DVM
    # that MPI_Comm_spawn starts inherits this process's environment, so this is
    # the environment that decides whether PRRTE will fork as root and which
    # hostname PMIx hands out. The shell that launched mpirun dumps the same
    # variables; a difference between the two means mpirun dropped them.
    mpi_env = {
        k: v
        for k, v in sorted(os.environ.items())
        if k.startswith(("PRTE_", "OMPI_", "PMIX_", "PMI_"))
    }
    print(
        f"[pid {os.getpid()}] MPI environment in rank process: "
        f"{mpi_env or '(none set)'}",
        flush=True)

    tasks = [0]
    assert os.environ[
        'TLLM_SPAWN_PROXY_PROCESS_IPC_ADDR'] is not None, "TLLM_SPAWN_PROXY_PROCESS_IPC_ADDR is not set"
    hmac_key = get_spawn_proxy_process_ipc_hmac_key_env()
    client = RemoteMpiCommSessionClient(
        os.environ['TLLM_SPAWN_PROXY_PROCESS_IPC_ADDR'], hmac_key=hmac_key)
    for task in tasks:
        if task_type == "submit":
            client.submit(print_colored, f"{task}\n", "green")
        elif task_type in ("submit_sync", "flashinfer_temporary_cleanup"):
            res = client.submit_sync(print_colored, f"{task}\n", "green")
            print(res)
        elif task_type == "flashinfer_workspace":
            workspaces = set(
                client.submit_sync(os.getenv, "FLASHINFER_WORKSPACE_BASE"))
            cubin_dirs = set(
                client.submit_sync(os.getenv, "FLASHINFER_CUBIN_DIR"))
            assert None not in workspaces
            assert len(workspaces) == 2
            workspace_root = (Path.home() / ".cache" / "tensorrt_llm" /
                              "flashinfer")
            assert all(
                Path(workspace).parent == workspace_root
                for workspace in workspaces)
            # Unset means FlashInfer derives the artifact cache from each
            # worker's isolated workspace, keeping downloaded compiler inputs
            # per-rank.
            assert cubin_dirs == {None}

            nested_session = MpiPoolSession(n_workers=2)
            try:
                nested_worker_envs = nested_session.submit_sync(
                    get_flashinfer_environment)
            finally:
                nested_session.shutdown()
            nested_workspaces = {
                workspace
                for workspace, _ in nested_worker_envs
            }
            assert None not in nested_workspaces
            assert len(nested_workspaces) == 2
            assert all(
                Path(workspace).parent == workspace_root
                for workspace in nested_workspaces)
            nested_cubin_dirs = {
                cubin_dir
                for _, cubin_dir in nested_worker_envs
            }
            assert nested_cubin_dirs == {None}
        elif task_type == "async_rank_failure":
            # Rank 1 fails at once while rank 0 is still running. The failure
            # must reach the client well before rank 0's task ends: with a
            # collective after the task, the failed rank would wait there for
            # rank 0 and the client would learn nothing until then.
            hold_seconds = 20.0
            client.submit(fail_on_rank_1_while_rank_0_runs, hold_seconds)
            deadline = time.monotonic() + hold_seconds / 2
            while (error := client.check_worker_error()) is None:
                assert time.monotonic() < deadline, (
                    "rank failure was not reported while its peer was still "
                    "running")
                time.sleep(0.5)
            assert RANK_1_FAILURE_MESSAGE in str(error), error
            print(f"rank failure reported to the client: {error}")


if __name__ == "__main__":
    main()
