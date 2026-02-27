# SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from typing import List

import torch
from strenum import StrEnum
from torch.distributed import ProcessGroup

from tensorrt_llm._torch.device_mesh import DeviceMeshTopologyImpl
from tensorrt_llm._utils import mpi_disabled


class CpType(StrEnum):
    # CP type for ulysses parallelism
    ULYSSES = "ULYSSES"
    # CP type for star attention
    STAR = "STAR"
    # CP type for ring attention
    RING = "RING"
    # CP type for helix parallelism
    HELIX = "HELIX"


class MappingBase:
    """Base class for distributed mapping configurations"""

    tp_rank: int
    pp_rank: int
    cp_rank: int

    def __init__(
            self,
            world_size=1,
            rank=0,
            gpus_per_node=8,
            *,
            cp_size=1,
            cp_config=None,
            tp_size=1,
            pp_size=1,
            pp_partition=None,
            moe_cluster_size=-1,  # -1 means no moe
            moe_tp_size=-1,  # -1 means no moe
            moe_ep_size=-1,  # -1 means no moe
            attn_tp_size=-1,
            attn_cp_size=-1,
            enable_attention_dp=False,
            enable_lm_head_tp_in_adp=False):
        # set default values for non-moe cases
        # or where only one MOE parallelism size is specified
        if moe_cluster_size == -1:
            moe_cluster_size = 1

        # Set default cp_type to ULYSSES.
        cp_type = CpType.ULYSSES

        # Convert cp_type to CpType enum if it is a string.
        if cp_config is not None:
            if "cp_type" in cp_config and isinstance(cp_config["cp_type"], str):
                cp_type_str = cp_config["cp_type"].upper()
                if cp_type_str not in CpType.__members__:
                    raise ValueError(
                        f"Invalid cp_type: {cp_config['cp_type']}. "
                        f"Must be one of: {', '.join([t.name for t in CpType])}"
                    )
                cp_config["cp_type"] = CpType(cp_type_str)
            cp_type = cp_config.get("cp_type", CpType.ULYSSES)

        moe_world_size = tp_size if cp_type == CpType.ULYSSES else tp_size * cp_size

        if moe_tp_size == -1 and moe_ep_size == -1:
            moe_tp_size = moe_world_size // moe_cluster_size
            moe_ep_size = 1

        elif moe_tp_size == -1:
            moe_tp_size = moe_world_size // (moe_ep_size * moe_cluster_size)

        elif moe_ep_size == -1:
            moe_ep_size = moe_world_size // (moe_tp_size * moe_cluster_size)

        if attn_tp_size == -1 and attn_cp_size == -1:
            if cp_type == CpType.ULYSSES:
                # fallback to ulysses
                attn_tp_size = tp_size * cp_size
                attn_cp_size = 1
            else:
                # fallback to helix
                attn_tp_size = tp_size
                attn_cp_size = cp_size

        elif attn_tp_size == -1:
            attn_tp_size = (tp_size * cp_size) // attn_cp_size

        elif attn_cp_size == -1:
            attn_cp_size = (tp_size * cp_size) // attn_tp_size

        if attn_cp_size != 1 and cp_type == CpType.ULYSSES:
            raise ValueError(
                f"attn_cp_size must be 1 for now for ulysses, but got {attn_tp_size}, {attn_cp_size}."
            )

        if tp_size * pp_size * cp_size != world_size:
            raise ValueError(
                "world_size must equal to tp_size * pp_size * cp_size, "
                f"but got {world_size} != {tp_size} * {pp_size} * {cp_size}.")

        moe_tp_ep_size = moe_tp_size * moe_ep_size
        self.moe_tp_cluster_ep_size = moe_tp_ep_size * moe_cluster_size
        if self.moe_tp_cluster_ep_size != moe_world_size:
            raise ValueError(
                "moe_tp_size * moe_ep_size * moe_cluster_size must equal to moe_world_size, "
                f"but got {self.moe_tp_cluster_ep_size} != {moe_world_size}")

        attn_tp_cp_size = attn_tp_size * attn_cp_size
        if attn_tp_cp_size != tp_size * cp_size:
            raise ValueError(
                "tp_size * cp_size must equal to attn_tp_size * attn_cp_size, "
                f"but got {tp_size} * {cp_size} != {attn_tp_size} * {attn_cp_size}"
            )

        if moe_ep_size != 1 and cp_size > 1 and cp_type != CpType.HELIX:
            raise NotImplementedError(
                f"CP {cp_type} doesn't support MoE tp/ep yet")

        if moe_cluster_size > 1:
            assert moe_ep_size == 1

        self.tp_size = tp_size
        self.cp_size = cp_size
        self.cp_config = cp_config if cp_config is not None else {}
        self.pp_size = pp_size
        self.pp_partition = pp_partition
        self.moe_tp_size = moe_tp_size
        self.moe_ep_size = moe_ep_size
        self.moe_cluster_size = moe_cluster_size
        self.attn_tp_size = attn_tp_size
        self.attn_cp_size = attn_cp_size
        self.world_size = world_size
        self.enable_attention_dp = enable_attention_dp
        if enable_lm_head_tp_in_adp:
            assert enable_attention_dp, "enable_lm_head_tp_in_adp requires enable_attention_dp"
        self.enable_lm_head_tp_in_adp = enable_lm_head_tp_in_adp
        self.rank = rank
        self.gpus_per_node = gpus_per_node

        self.pp_groups = []
        self.cp_groups = []
        self.tp_groups = []
        self.moe_cluster_groups = []
        self.moe_tp_groups = []
        self.moe_ep_groups = []

    def __eq__(self, other):
        if not isinstance(other, MappingBase):
            return NotImplemented

        return (self.world_size == other.world_size and self.rank == other.rank
                and self.gpus_per_node == other.gpus_per_node
                and self.cp_size == other.cp_size
                and self.tp_size == other.tp_size
                and self.moe_cluster_size == other.moe_cluster_size
                and self.pp_size == other.pp_size
                and self.pp_partition == other.pp_partition
                and self.moe_tp_size == other.moe_tp_size
                and self.moe_ep_size == other.moe_ep_size
                and self.attn_tp_size == other.attn_tp_size
                and self.attn_cp_size == other.attn_cp_size
                and self.cp_config == other.cp_config)

    def __hash__(self):
        return hash((
            self.world_size,
            self.rank,
            self.gpus_per_node,
            self.cp_size,
            self.tp_size,
            self.pp_size,
            self.moe_tp_size,
            self.moe_cluster_size,
            self.moe_ep_size,
            self.attn_tp_size,
            self.attn_cp_size,
            # note: we do not allow updating cp_config after initialization
            tuple(sorted(self.cp_config.items())),
            tuple(self.pp_partition) if self.pp_partition is not None else (),
        ))

    @property
    def rank(self):
        return self._rank

    @rank.setter
    def rank(self, rank: int):
        # TODO(qijun): skip check for enable_attention_dp temporarily, will support attention_dp_size
        if not self.enable_attention_dp:
            if not isinstance(rank, int) or rank < 0 or rank >= self.world_size:
                raise ValueError(
                    f"Rank should be an integer between 0 and {self.world_size-1}, but got {rank}."
                )
        self._rank = rank

    @property
    def moe_tp_rank(self):
        return self.tp_rank // (self.moe_ep_size * self.moe_cluster_size)

    @property
    def moe_cluster_rank(self):
        return self.tp_rank % self.moe_cluster_size

    @property
    def moe_ep_rank(self):
        return self.tp_rank % self.moe_ep_size

    @property
    def moe_cluster_group(self):
        return self.moe_cluster_groups[self.pp_rank * self.moe_tp_size +
                                       self.moe_tp_rank]

    @property
    def node_rank(self):
        return self.rank // self.gpus_per_node

    @property
    def local_rank(self):
        return self.rank % self.gpus_per_node

    @property
    def dp_size(self):
        return self.tp_size if self.enable_attention_dp else 1

    def has_cp_ulysses(self):
        return self.cp_size > 1 and self.cp_config.get(
            "cp_type") == CpType.ULYSSES

    def has_cp_helix(self):
        return self.cp_size > 1 and self.cp_config.get(
            "cp_type") == CpType.HELIX

    def get_node_rank(self, rank: int):
        return rank // self.gpus_per_node

    def get_local_rank(self, rank: int):
        return rank % self.gpus_per_node

    def is_multi_node(self):
        return self.world_size > self.gpus_per_node

    def has_tp(self):
        return self.tp_size > 1

    def is_last_pp_rank(self):
        return self.pp_rank == self.pp_size - 1

    def is_second_last_pp_rank(self):
        return self.pp_rank == self.pp_size - 2

    def is_first_pp_rank(self):
        return self.pp_rank == 0

    def has_pp(self):
        return self.pp_size > 1

    def prev_pp_rank(self):
        p = self.rank - self.tp_size * self.cp_size
        if p < 0:
            p = p + self.world_size
        return p

    def next_pp_rank(self):
        p = self.rank + self.tp_size * self.cp_size
        if p >= self.world_size:
            p = p - self.world_size
        return p

    def is_last_cp_rank(self):
        return self.cp_rank == self.cp_size - 1

    def is_first_cp_rank(self):
        return self.cp_rank == 0

    def has_cp(self):
        return self.cp_size > 1

    def prev_cp_rank(self):
        # cp ranks are consecutive, so prev is rank - 1 with wraparound within cp group.
        if self.cp_rank == 0:
            return self.rank + self.cp_size - 1
        return self.rank - 1

    def next_cp_rank(self):
        # cp ranks are consecutive, so next is rank + 1 with wraparound within cp group.
        if self.cp_rank == self.cp_size - 1:
            return self.rank - self.cp_size + 1
        return self.rank + 1

    def has_moe_cluster(self):
        return self.moe_cluster_size > 1

    def has_moe_tp(self):
        return self.moe_tp_size > 1

    def has_moe_ep(self):
        return self.moe_ep_size > 1

    def pp_layers(self, num_layers: int) -> List[int]:
        if self.pp_partition is not None:
            if len(self.pp_partition) != self.pp_size:
                raise ValueError(
                    f"{len(self.pp_partition)=} does not match {self.pp_size=}."
                )
            if sum(self.pp_partition) != num_layers:
                raise ValueError(
                    f"{sum(self.pp_partition)=} does not match {num_layers=}.")
            return torch.arange(num_layers).split(
                self.pp_partition)[self.pp_rank].tolist()
        else:
            # If num_layers % pp_size = n != 0, first n ranks get one extra layer
            return torch.tensor_split(torch.arange(num_layers),
                                      self.pp_size)[self.pp_rank].tolist()

    def pp_rank_of_layer(self, layer_idx: int, num_layers: int) -> int:
        """Return pipeline-parallel rank that owns `layer_idx` for a model with `num_layers` layers.
        Mirrors the partitioning behavior in `pp_layers()`.
        """
        if layer_idx < 0 or layer_idx >= num_layers:
            raise ValueError(f"{layer_idx=} is out of range for {num_layers=}.")
        if not self.has_pp():
            return 0

        if self.pp_partition is not None:
            if len(self.pp_partition) != self.pp_size:
                raise ValueError(
                    f"{len(self.pp_partition)=} does not match {self.pp_size=}."
                )
            if sum(self.pp_partition) != num_layers:
                raise ValueError(
                    f"{sum(self.pp_partition)=} does not match {num_layers=}.")
            end = 0
            for pp_rank, n in enumerate(self.pp_partition):
                end += n
                if layer_idx < end:
                    return pp_rank
            raise RuntimeError("Unreachable: invalid pp_partition.")

        base, rem = divmod(num_layers, self.pp_size)
        if base == 0:
            # Matches torch.tensor_split: first `num_layers` ranks get one layer.
            return layer_idx
        cutoff = (base + 1) * rem
        if layer_idx < cutoff:
            return layer_idx // (base + 1)
        return rem + (layer_idx - cutoff) // base

    def ep_experts(self, num_experts: int) -> List[int]:
        assert self.cp_size == 1
        experts_per_rank = num_experts // self.moe_ep_size
        experts_range = range(self.moe_ep_rank * experts_per_rank,
                              (self.moe_ep_rank + 1) * experts_per_rank)
        return list(experts_range)

    @classmethod
    def from_dict(cls, mapping: dict):
        return cls(**mapping)

    def to_dict(self):
        return {
            'world_size': self.world_size,
            'rank': self.rank,
            'gpus_per_node': self.gpus_per_node,
            'cp_size': self.cp_size,
            'tp_size': self.tp_size,
            'pp_size': self.pp_size,
            'moe_tp_size': self.moe_tp_size,
            'moe_cluster_size': self.moe_cluster_size,
            'moe_ep_size': self.moe_ep_size,
            'attn_tp_size': self.attn_tp_size,
            'attn_cp_size': self.attn_cp_size,
            'cp_config': self.cp_config,
            'enable_attention_dp': self.enable_attention_dp,
            'enable_lm_head_tp_in_adp': self.enable_lm_head_tp_in_adp,
        }


class Mapping(MappingBase):
    """
    A node with 8 GPUs, tp_size = 4, cp_size = 1, pp_size = 2

    2 tp groups:

    - [0, 1, 2, 3]
    - [4, 5, 6, 7]

    4 pp groups:

    - [0, 4]
    - [1, 5]
    - [2, 6]
    - [3, 7]

    A node with 8 GPUs, tp_size = 4, cp_size = 2, pp_size = 1

    4 cp groups:

    - [0, 1]
    - [2, 3]
    - [4, 5]
    - [6, 7]

    2 tp groups:

    - [0, 2, 4, 6]
    - [1, 3, 5, 7]

    A node with 8 GPUs, moe_tp_size = 2, moe_ep_size = 4

    4 moe_tp groups:

    - [0, 4]
    - [1, 5]
    - [2, 6]
    - [3, 7]

    2 moe_ep groups:

    - [0, 1, 2, 3]
    - [4, 5, 6, 7]

    2 nodes with 16 GPUs, moe_tp_size = 2, moe_ep_size = 4, pp_size = 2

    8 moe_tp groups:

    - [0 4]
    - [1 5]
    - [2 6]
    - [3 7]
    - [8 12]
    - [9 13]
    - [10 14]
    - [11 15]

    4 moe_ep groups:

    - [0, 1, 2, 3]
    - [4, 5, 6, 7]
    - [8, 9, 10, 11]
    - [12, 13, 14, 15]

    8 pp groups:

    - [0 8]
    - [1 9]
    - [2 10]
    - [3 11]
    - [4 12]
    - [5 13]
    - [6 14]
    - [7 15]

    2 nodes with 8 GPUs, tp_size 2, pp_size 2, cp_size 2

    4 cp groups:
    - [0, 1]
    - [2, 3]
    - [4, 5]
    - [6, 7]

    4 tp groups:
    - [0, 2]
    - [1, 3]
    - [4, 6]
    - [5, 7]

    4 pp groups:
    - [0, 4]
    - [1, 5]
    - [2, 6]
    - [3, 7]
    """

    def __new__(cls, *args, **kwargs):
        if mpi_disabled():
            return super().__new__(DeviceMeshTopology)
        else:
            return super().__new__(MpiTopology)

    # Intentionally repeated for type hints
    def __init__(
            self,
            world_size=1,
            rank=0,
            gpus_per_node=8,
            *,
            cp_size=1,
            cp_config=None,
            tp_size=1,
            pp_size=1,
            pp_partition=None,
            moe_cluster_size=-1,  # -1 means no moe
            moe_tp_size=-1,  # -1 means no moe
            moe_ep_size=-1,  # -1 means no moe
            attn_tp_size=-1,
            attn_cp_size=-1,
            enable_attention_dp=False,
            enable_lm_head_tp_in_adp=False):
        super().__init__(world_size=world_size,
                         rank=rank,
                         gpus_per_node=gpus_per_node,
                         cp_size=cp_size,
                         cp_config=cp_config,
                         tp_size=tp_size,
                         pp_size=pp_size,
                         pp_partition=pp_partition,
                         moe_cluster_size=moe_cluster_size,
                         moe_tp_size=moe_tp_size,
                         moe_ep_size=moe_ep_size,
                         attn_tp_size=attn_tp_size,
                         attn_cp_size=attn_cp_size,
                         enable_attention_dp=enable_attention_dp,
                         enable_lm_head_tp_in_adp=enable_lm_head_tp_in_adp)

    def repurpose_helix_cp_to_tp(self):
        # In helix parallelism, CP is relevant only for the attention layer. These ranks are repurposed to TP
        # for FFN layers.
        assert self.has_cp_helix()
        return Mapping(
            world_size=self.world_size,
            rank=self.rank,
            gpus_per_node=self.gpus_per_node,
            cp_size=1,
            cp_config={},
            tp_size=self.tp_size * self.cp_size,
            pp_size=self.pp_size,
            pp_partition=self.pp_partition,
            moe_cluster_size=self.moe_cluster_size,
            moe_tp_size=self.moe_tp_size,
            moe_ep_size=self.moe_ep_size,
            # attn_tp_size, attn_cp_size shall be set in the constructor of Mapping.
            enable_attention_dp=self.enable_attention_dp,
            enable_lm_head_tp_in_adp=self.enable_lm_head_tp_in_adp)

    # DeviceMesh specific methods
    @property
    def tp_group_pg(self) -> ProcessGroup:
        raise NotImplementedError("tp_group_pg is not implemented.")

    @property
    def pp_group_pg(self) -> ProcessGroup:
        raise NotImplementedError("pp_group_pg is not implemented.")

    @property
    def cp_group_pg(self) -> ProcessGroup:
        raise NotImplementedError("cp_group_pg is not implemented.")

    @property
    def moe_tp_group_pg(self) -> ProcessGroup:
        raise NotImplementedError("moe_tp_group_pg is not implemented.")

    @property
    def moe_ep_group_pg(self) -> ProcessGroup:
        raise NotImplementedError("moe_ep_group_pg is not implemented.")

    def build_mesh(self):
        raise NotImplementedError("build_mesh is not implemented.")


class MpiTopology(Mapping):
    '''MPI-based mapping implementation'''

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._init_parallel_groups()

    @property
    def tp_rank(self) -> int:
        return self.rank % (self.tp_size * self.cp_size) // self.cp_size

    @property
    def pp_rank(self) -> int:
        return self.rank // (self.tp_size * self.cp_size)

    @property
    def cp_rank(self) -> int:
        return self.rank % self.cp_size

    @property
    def tp_group(self) -> List[int]:
        return self.tp_groups[self.pp_rank * self.cp_size + self.cp_rank]

    @property
    def pp_group(self) -> List[int]:
        return self.pp_groups[self.tp_rank * self.cp_size + self.cp_rank]

    @property
    def cp_group(self) -> List[int]:
        return self.cp_groups[self.pp_rank * self.tp_size + self.tp_rank]

    @property
    def moe_tp_group(self) -> List[int]:
        return self.moe_tp_groups[self.pp_rank * self.moe_cluster_size *
                                  self.moe_ep_size +
                                  self.moe_cluster_rank * self.moe_ep_size +
                                  self.moe_ep_rank]

    @property
    def moe_ep_group(self) -> List[int]:
        return self.moe_ep_groups[self.pp_rank * self.moe_tp_size *
                                  self.moe_cluster_size +
                                  self.moe_tp_rank * self.moe_cluster_size +
                                  self.moe_cluster_rank]

    @property
    def moe_cluster_group(self) -> List[int]:
        return self.moe_cluster_groups[self.pp_rank * self.moe_tp_size +
                                       self.moe_tp_rank]

    def _init_parallel_groups(self):
        # init pp group
        for i in range(self.tp_size * self.cp_size):
            ranks = range(i, self.world_size, self.tp_size * self.cp_size)
            self.pp_groups.append(list(ranks))

        # init cp group (consecutive ranks within each tp slice).
        for i in range(self.pp_size):
            for j in range(self.tp_size):
                ranks = range(
                    i * self.tp_size * self.cp_size + j * self.cp_size,
                    i * self.tp_size * self.cp_size + (j + 1) * self.cp_size)
                self.cp_groups.append(list(ranks))

        # init tp group (interleaved ranks with stride of cp_size).
        for i in range(self.pp_size):
            for j in range(self.cp_size):
                ranks = range(i * self.tp_size * self.cp_size + j,
                              (i + 1) * self.tp_size * self.cp_size + j,
                              self.cp_size)
                self.tp_groups.append(list(ranks))

        # init moe tp group
        for i in range(self.pp_size):
            for j in range(self.moe_cluster_size * self.moe_ep_size):
                ranks = range(i * self.moe_tp_cluster_ep_size + j,
                              (i + 1) * self.moe_tp_cluster_ep_size,
                              self.moe_cluster_size * self.moe_ep_size)
                self.moe_tp_groups.append(list(ranks))

        # init moe cluster group
        for i in range(self.pp_size):
            for j in range(self.moe_tp_size):
                ranks = range(
                    i * self.moe_tp_cluster_ep_size +
                    j * self.moe_cluster_size * self.moe_ep_size,
                    i * self.moe_tp_cluster_ep_size +
                    (j + 1) * self.moe_cluster_size * self.moe_ep_size)
                self.moe_cluster_groups.append(list(ranks))

        # init moe ep group
        for i in range(self.pp_size):
            for j in range(self.moe_tp_size):
                for k in range(self.moe_cluster_size):
                    ranks = range(
                        i * self.moe_tp_cluster_ep_size +
                        j * self.moe_cluster_size * self.moe_ep_size +
                        k * self.moe_ep_size, i * self.moe_tp_cluster_ep_size +
                        j * self.moe_cluster_size * self.moe_ep_size +
                        (k + 1) * self.moe_ep_size)
                    self.moe_ep_groups.append(list(ranks))


class DeviceMeshTopology(DeviceMeshTopologyImpl, Mapping):
    """PyTorch DeviceMesh-based mapping implementation"""

    def __init__(self, *args, **kwargs):
        assert mpi_disabled(
        ), "DeviceMeshTopology is only available in Ray orchestrator mode."

        super().__init__(*args, **kwargs)


class DraftMapping:
    """Mapping adapter for draft models with different TP in 1-model speculative decoding.

    The draft model runs on the same GPUs as the target model but uses smaller
    TP sub-groups. The target's TP group is split into independent sub-groups,
    each running a replica of the draft model.

    For example, target TP=8 with draft_tp_size=2 creates four independent
    draft TP sub-groups: [0,1], [2,3], [4,5], [6,7].

    ProcessGroups for the sub-groups are created collectively during
    construction, so all ranks must instantiate this class simultaneously.
    """

    def __init__(self,
                 target_mapping: Mapping,
                 draft_tp_size: int,
                 draft_moe_tp_size: int = -1,
                 draft_moe_ep_size: int = -1):
        if target_mapping.tp_size % draft_tp_size != 0:
            raise ValueError(
                f"Target TP size ({target_mapping.tp_size}) must be divisible "
                f"by draft TP size ({draft_tp_size})")
        if draft_tp_size < 1:
            raise ValueError(
                f"Draft TP size must be >= 1, got {draft_tp_size}")

        self._target = target_mapping
        self._draft_tp_size = draft_tp_size

        target_tp_group = target_mapping.tp_group
        target_tp_rank = target_mapping.tp_rank

        num_sub_groups = target_mapping.tp_size // draft_tp_size
        sub_groups = [
            target_tp_group[i * draft_tp_size:(i + 1) * draft_tp_size]
            for i in range(num_sub_groups)
        ]

        group_idx = target_tp_rank // draft_tp_size
        self._tp_rank = target_tp_rank % draft_tp_size
        self._tp_group = sub_groups[group_idx]

        if draft_moe_tp_size == -1 and draft_moe_ep_size == -1:
            self._moe_tp_size = draft_tp_size
            self._moe_ep_size = 1
        elif draft_moe_tp_size == -1:
            self._moe_ep_size = draft_moe_ep_size
            self._moe_tp_size = draft_tp_size // draft_moe_ep_size
        elif draft_moe_ep_size == -1:
            self._moe_tp_size = draft_moe_tp_size
            self._moe_ep_size = draft_tp_size // draft_moe_tp_size
        else:
            self._moe_tp_size = draft_moe_tp_size
            self._moe_ep_size = draft_moe_ep_size

        if self._moe_tp_size * self._moe_ep_size != draft_tp_size:
            raise ValueError(
                f"draft moe_tp_size * moe_ep_size ({self._moe_tp_size} * "
                f"{self._moe_ep_size}) must equal draft_tp_size "
                f"({draft_tp_size})")

        self._moe_tp_rank = self._tp_rank // self._moe_ep_size
        self._moe_ep_rank = self._tp_rank % self._moe_ep_size

        draft_tp_sub = self._tp_group
        moe_tp_sub_groups = [
            draft_tp_sub[j::self._moe_ep_size]
            for j in range(self._moe_ep_size)
        ]
        moe_ep_sub_groups = [
            draft_tp_sub[j * self._moe_ep_size:(j + 1) * self._moe_ep_size]
            for j in range(self._moe_tp_size)
        ]
        self._moe_tp_group = moe_tp_sub_groups[self._moe_ep_rank]
        self._moe_ep_group = moe_ep_sub_groups[self._moe_tp_rank]

        self._tp_pg = None
        self._moe_tp_pg = None
        self._moe_ep_pg = None
        self._create_process_groups(sub_groups)

    def _create_process_groups(self, tp_sub_groups):
        try:
            import torch.distributed as dist
            if not dist.is_initialized():
                return

            for sg in tp_sub_groups:
                pg = dist.new_group(sg)
                if self._target.rank in sg:
                    self._tp_pg = pg

            all_moe_tp_groups = set()
            all_moe_ep_groups = set()
            for sg in tp_sub_groups:
                for j in range(self._moe_ep_size):
                    all_moe_tp_groups.add(tuple(sg[j::self._moe_ep_size]))
                for j in range(self._moe_tp_size):
                    all_moe_ep_groups.add(
                        tuple(sg[j * self._moe_ep_size:(j + 1) *
                                 self._moe_ep_size]))

            for grp in sorted(all_moe_tp_groups):
                pg = dist.new_group(list(grp))
                if self._target.rank in grp:
                    self._moe_tp_pg = pg

            for grp in sorted(all_moe_ep_groups):
                pg = dist.new_group(list(grp))
                if self._target.rank in grp:
                    self._moe_ep_pg = pg
        except ImportError:
            pass

    @property
    def tp_size(self):
        return self._draft_tp_size

    @property
    def tp_rank(self):
        return self._tp_rank

    @property
    def tp_group(self):
        return self._tp_group

    @property
    def tp_group_pg(self):
        return self._tp_pg

    @property
    def moe_tp_size(self):
        return self._moe_tp_size

    @property
    def moe_ep_size(self):
        return self._moe_ep_size

    @property
    def moe_tp_rank(self):
        return self._moe_tp_rank

    @property
    def moe_ep_rank(self):
        return self._moe_ep_rank

    @property
    def moe_tp_group(self):
        return self._moe_tp_group

    @property
    def moe_ep_group(self):
        return self._moe_ep_group

    @property
    def moe_tp_group_pg(self):
        return self._moe_tp_pg

    @property
    def moe_ep_group_pg(self):
        return self._moe_ep_pg

    @property
    def moe_cluster_size(self):
        return 1

    @property
    def moe_cluster_rank(self):
        return 0

    @property
    def pp_size(self):
        return self._target.pp_size

    @property
    def pp_rank(self):
        return self._target.pp_rank

    @property
    def cp_size(self):
        return self._target.cp_size

    @property
    def cp_rank(self):
        return self._target.cp_rank

    @property
    def rank(self):
        return self._target.rank

    @property
    def world_size(self):
        return self._target.world_size

    @property
    def gpus_per_node(self):
        return self._target.gpus_per_node

    @property
    def local_rank(self):
        return self._target.local_rank

    @property
    def node_rank(self):
        return self._target.node_rank

    @property
    def enable_attention_dp(self):
        return self._target.enable_attention_dp

    @property
    def enable_lm_head_tp_in_adp(self):
        return self._target.enable_lm_head_tp_in_adp

    @property
    def dp_size(self):
        return self._draft_tp_size if self.enable_attention_dp else 1

    @property
    def moe_tp_cluster_ep_size(self):
        return self._moe_tp_size * self._moe_ep_size

    def has_tp(self):
        return self._draft_tp_size > 1

    def has_pp(self):
        return self._target.has_pp()

    def has_cp(self):
        return self._target.has_cp()

    def has_moe_tp(self):
        return self._moe_tp_size > 1

    def has_moe_ep(self):
        return self._moe_ep_size > 1

    def has_moe_cluster(self):
        return False

    def get_local_rank(self, rank: int):
        return self._target.get_local_rank(rank)

    def get_node_rank(self, rank: int):
        return self._target.get_node_rank(rank)

    def ep_experts(self, num_experts: int):
        experts_per_rank = num_experts // self._moe_ep_size
        start = self._moe_ep_rank * experts_per_rank
        return list(range(start, start + experts_per_rank))

    @property
    def cp_config(self):
        return self._target.cp_config

    @property
    def attn_tp_size(self):
        return self._draft_tp_size

    @property
    def attn_cp_size(self):
        return 1

    def has_cp_ulysses(self):
        return self._target.has_cp_ulysses()

    def has_cp_helix(self):
        return self._target.has_cp_helix()

    def is_multi_node(self):
        return self._target.is_multi_node()

    def pp_layers(self, num_layers: int):
        return self._target.pp_layers(num_layers)

    def is_last_pp_rank(self):
        return self._target.is_last_pp_rank()

    def is_first_pp_rank(self):
        return self._target.is_first_pp_rank()

    def __eq__(self, other):
        if not isinstance(other, DraftMapping):
            return NotImplemented
        return (self._draft_tp_size == other._draft_tp_size
                and self._moe_tp_size == other._moe_tp_size
                and self._moe_ep_size == other._moe_ep_size
                and self._target.rank == other._target.rank
                and tuple(self._tp_group) == tuple(other._tp_group))

    def __hash__(self):
        return hash(
            (self._draft_tp_size, self._moe_tp_size, self._moe_ep_size,
             self._target.rank, tuple(self._tp_group)))
