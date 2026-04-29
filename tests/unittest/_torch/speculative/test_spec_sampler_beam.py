import pytest
import torch

from tensorrt_llm._torch.pyexecutor.llm_request import FinishReason, LlmRequestState
from tensorrt_llm._torch.pyexecutor.sampler import TorchSampler
from tensorrt_llm._torch.speculative.mtp import MTPSampler
from tensorrt_llm._torch.speculative.spec_sampler_base import SampleStateSpec, SampleStateTensorsSpec


class _FakeSamplerEvent:

    def synchronize(self) -> None:
        return


class _FakeSamplingConfig:

    def __init__(self, beam_width: int):
        self.beam_width = beam_width


class _FakeRequest:

    def __init__(self, *, seq_slot: int, beam_width: int):
        self.py_seq_slot = seq_slot
        self.py_request_id = 0
        self.state = LlmRequestState.GENERATION_IN_PROGRESS
        self.sampling_config = _FakeSamplingConfig(beam_width)
        self.py_return_context_logits = False
        self.py_return_generation_logits = False
        self.py_return_log_probs = False
        self.py_draft_tokens = []
        self.py_decoding_iter = 0
        self.py_end_id = -1
        self.py_orig_prompt_len = 0
        self.py_max_new_tokens = 128
        self.py_stop_words_list = None
        self.py_num_accepted_draft_tokens = 0
        self.py_rewind_len = 0
        self._tokens = [[] for _ in range(beam_width)]
        self._finished_reasons = [FinishReason.NOT_FINISHED for _ in range(beam_width)]

    def add_new_token(self, new_token: int, beam_idx: int) -> None:
        self._tokens[beam_idx].append(new_token)

    def get_num_tokens(self, beam_idx: int) -> int:
        return len(self._tokens[beam_idx])

    def get_tokens(self, beam_idx: int):
        return self._tokens[beam_idx]

    def set_finished_reason(self, reason: FinishReason, beam_idx: int) -> None:
        self._finished_reasons[beam_idx] = reason

    def get_finished_reason(self, beam_idx: int) -> FinishReason:
        return self._finished_reasons[beam_idx]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_mtp_sampler_accepts_beam_width_gt_one():
    args = TorchSampler.Args(
        max_seq_len=16,
        max_draft_len=2,
        max_total_draft_tokens=2,
        max_num_sequences=4,
        max_beam_width=2,
    )
    sampler = MTPSampler(args, nextn=2)
    assert sampler.max_beam_width == 2
    assert sampler.store.new_tokens.shape[-1] == 2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_spec_sampler_update_requests_updates_each_beam():
    args = TorchSampler.Args(
        max_seq_len=16,
        max_draft_len=2,
        max_total_draft_tokens=2,
        max_num_sequences=2,
        max_beam_width=2,
    )
    sampler = MTPSampler(args, nextn=2)
    req = _FakeRequest(seq_slot=0, beam_width=2)

    host_tensors = SampleStateTensorsSpec(
        new_tokens=torch.tensor(
            [
                [[11, 21]],
                [[12, 22]],
                [[13, 23]],
            ],
            dtype=torch.int32,
            device="cpu",
        ),
        new_tokens_lens=torch.tensor([2], dtype=torch.int32, device="cpu"),
        next_draft_tokens=torch.tensor([[31, 32]], dtype=torch.int32, device="cpu"),
    )
    state = SampleStateSpec(
        requests=[req],
        host=host_tensors,
        device=host_tensors,
        sampler_event=_FakeSamplerEvent(),
        runtime_draft_len=2,
    )

    sampler.update_requests(state)

    assert req.get_tokens(0) == [11, 12]
    assert req.get_tokens(1) == [21, 22]
    assert req.py_num_accepted_draft_tokens == 1
    assert req.py_rewind_len == 1
    assert req.py_draft_tokens == [31, 32]
