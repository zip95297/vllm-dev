# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for SuffixSpeculator (V2 model runner; requires CUDA +
suffix_gpu)."""

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("suffix_gpu")

from vllm.v1.worker.gpu.spec_decode.suffix.speculator import SuffixSpeculator
from vllm.v1.worker.gpu.states import RequestState

if not torch.cuda.is_available():
    pytest.skip("CUDA required", allow_module_level=True)

DEVICE = torch.device("cuda:0")
K = 8
MAX_NUM_SEQS = 8
MAX_MODEL_LEN = 256


def _make_config(use_cuda_graph: bool) -> SimpleNamespace:
    spec = SimpleNamespace(
        num_speculative_tokens=K,
        suffix_decoding_max_tree_depth=24,
        suffix_decoding_max_cached_requests=1000,
        suffix_decoding_max_spec_factor=2.0,
        suffix_decoding_min_token_prob=0.1,
        suffix_gpu_global_capacity=1 << 16,
        suffix_gpu_delta_capacity=1 << 12,
        suffix_gpu_max_occurrences=32,
        suffix_gpu_num_backoff=4,
        suffix_gpu_use_cuda_graph=use_cuda_graph,
        suffix_gpu_ingest_chunk=16,
    )
    return SimpleNamespace(
        speculative_config=spec,
        model_config=SimpleNamespace(max_model_len=MAX_MODEL_LEN),
        scheduler_config=SimpleNamespace(max_num_seqs=MAX_NUM_SEQS),
    )


def _make_speculator(use_cuda_graph: bool) -> SuffixSpeculator:
    speculator = SuffixSpeculator(_make_config(use_cuda_graph), DEVICE)
    req_states = RequestState(
        max_num_reqs=MAX_NUM_SEQS,
        max_model_len=MAX_MODEL_LEN,
        max_num_batched_tokens=2048,
        num_speculative_steps=K,
        vocab_size=32000,
        device=DEVICE,
    )
    speculator.set_request_states(req_states)
    return speculator


def _add_request(speculator: SuffixSpeculator, req_id: str, tokens: list[int]) -> int:
    """Register a request whose full history is `tokens` (as if post_update
    had already appended this step's sampled token)."""
    req_states = speculator.req_states
    req_states.add_request(
        req_id=req_id,
        prompt_len=len(tokens),
        all_token_ids=tokens,
        num_computed_tokens=len(tokens) - 1,
        max_tokens=64,
    )
    req_states.apply_staged_writes()
    return req_states.req_id_to_index[req_id]


def _propose(speculator: SuffixSpeculator, req_ids: list[str]) -> torch.Tensor:
    req_states = speculator.req_states
    idx_mapping_np = [req_states.req_id_to_index[r] for r in req_ids]
    input_batch = SimpleNamespace(
        num_reqs=len(req_ids),
        req_ids=req_ids,
        idx_mapping_np=idx_mapping_np,
        idx_mapping=torch.tensor(idx_mapping_np, dtype=torch.int64, device=DEVICE),
    )
    num_sampled = torch.ones(len(req_ids), dtype=torch.int32, device=DEVICE)
    draft = speculator.propose(
        input_batch,
        attn_metadata={},
        slot_mappings={},
        last_hidden_states=None,
        aux_hidden_states=None,
        num_sampled=num_sampled,
        num_rejected=torch.zeros_like(num_sampled),
        last_sampled=req_states.last_sampled_tokens,
        next_prefill_tokens=req_states.next_prefill_tokens,
        temperature=None,
        seeds=None,
    )
    torch.accelerator.synchronize()
    return draft


def _raw_reference(tokens: list[int]) -> torch.Tensor:
    """Clamped drafter output for a single-row batch with `tokens` history."""
    from suffix_gpu.proposer import SuffixGPUDrafter

    cfg = _make_config(False).speculative_config
    drafter = SuffixGPUDrafter(
        k=K,
        device=DEVICE,
        max_pattern_len=cfg.suffix_decoding_max_tree_depth,
        min_match_len=1,
        max_occurrences=cfg.suffix_gpu_max_occurrences,
        enable_global=False,
        max_spec_factor=cfg.suffix_decoding_max_spec_factor,
        max_spec_offset=0.0,
        min_token_prob=cfg.suffix_decoding_min_token_prob,
        num_backoff=cfg.suffix_gpu_num_backoff,
    )
    buf = torch.zeros(1, MAX_MODEL_LEN, dtype=torch.int32, device=DEVICE)
    buf[0, : len(tokens)] = torch.tensor(tokens, dtype=torch.int32)
    counts = torch.tensor([len(tokens)], dtype=torch.int32, device=DEVICE)
    draft, num_valid = drafter.propose(counts, buf)
    torch.accelerator.synchronize()
    return draft.to(torch.int64).clamp_(min=0)[0], int(num_valid[0])


HIST = [5, 6, 7, 8] * 3 + [5]


@pytest.mark.parametrize("use_cuda_graph", [False, True])
def test_propose_matches_raw_drafter(use_cuda_graph):
    speculator = _make_speculator(use_cuda_graph)
    _add_request(speculator, "req-0", HIST)
    draft = _propose(speculator, ["req-0"])
    assert draft.shape == (1, K)
    assert draft.dtype == torch.int64

    expected, num_valid = _raw_reference(HIST)
    assert num_valid > 0
    assert expected[:num_valid].tolist() == ([6, 7, 8, 5] * 3)[:num_valid]
    assert torch.equal(draft[0], expected)
    # Padded slots are clamped for embedding safety.
    assert int(draft.min()) >= 0
    if use_cuda_graph:
        assert speculator._graph is not None


def test_graph_and_eager_agree():
    speculator_g = _make_speculator(True)
    _add_request(speculator_g, "req-0", HIST)
    d_g = _propose(speculator_g, ["req-0"])
    speculator_e = _make_speculator(False)
    _add_request(speculator_e, "req-0", HIST)
    d_e = _propose(speculator_e, ["req-0"])
    assert torch.equal(d_e, d_g)


def test_dummy_run_returns_zeros_without_state():
    speculator = _make_speculator(False)
    input_batch = SimpleNamespace(num_reqs=3)
    draft = speculator.propose(
        input_batch,
        attn_metadata={},
        slot_mappings={},
        last_hidden_states=None,
        aux_hidden_states=None,
        num_sampled=None,
        num_rejected=None,
        last_sampled=None,
        next_prefill_tokens=None,
        temperature=None,
        seeds=None,
        dummy_run=True,
    )
    assert draft.shape == (3, K)
    assert int(draft.abs().sum()) == 0


def test_ingest_and_cross_request_draft():
    speculator = _make_speculator(False)
    speculator.ingest_chunk = 1
    phrase = list(range(100, 108))
    resp = phrase * 4
    _add_request(speculator, "req-a", resp)
    # prompt_len=0 marks the whole history as response, so it is ingested.
    speculator.req_states.prompt_len.np[
        speculator.req_states.req_id_to_index["req-a"]
    ] = 0

    _propose(speculator, ["req-a"])  # triggers chunked ingestion
    speculator.on_requests_finished(["req-a"])
    torch.accelerator.synchronize()
    assert "req-a" not in speculator.drafter._ingested

    cur = [7, 9] + phrase[:6]
    _add_request(speculator, "req-b", cur)
    draft = _propose(speculator, ["req-b"])
    expect = (phrase[6:] + phrase * 2)[:K]
    n = len([t for t in draft[0].tolist() if t != 0])
    assert n > 0
    assert draft[0, :n].tolist() == expect[:n]

