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

import time

RANK_1_FAILURE_MESSAGE = "rank 1 failed during the task"


def fail_on_rank_1_while_rank_0_runs(hold_seconds: float) -> None:
    """Rank 1 fails at once; rank 0 keeps its task open for ``hold_seconds``."""
    # Keep importing this pickling helper from initializing MPI in the parent.
    from mpi4py import MPI

    if MPI.COMM_WORLD.rank == 1:
        raise RuntimeError(RANK_1_FAILURE_MESSAGE)
    time.sleep(hold_seconds)
