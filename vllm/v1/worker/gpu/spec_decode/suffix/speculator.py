# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GPU-resident suffix decoding speculator for the V2 model runner.

Unlike the V1 proposer (vllm/v1/spec_decode/suffix_proposer_gpu.py), this
speculator keeps no token mirror of its own: RequestState already holds
slot-indexed, device-visible token history (``all_token_ids``, UVA) and
per-slot lengths (``total_len``), both brought current by post_update()
before propose() runs. The drafter kernels therefore run over the full
slot space with a per-slot active mask, which keeps every shape
step-invariant and makes the optional CUDA graph a single capture.
"""

from collections.abc import Iterable
from typing import Any

import torch

from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.logger import init_logger
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.spec_decode.speculator import BaseSpeculator
from vllm.v1.worker.gpu.states import RequestState

logger = init_logger(__name__)


class SuffixSpeculator(BaseSpeculator):
    """Token-history drafter: no draft model, no hidden states."""

    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        config = vllm_config.speculative_config
        assert config is not None, "Speculative config must be set"

        # Lazy import so vLLM works without the SuffixGPU package.
        from suffix_gpu.proposer import SuffixGPUDrafter

        self.k = config.num_speculative_tokens
        self.max_model_len = vllm_config.model_config.max_model_len
        self.max_num_reqs = vllm_config.scheduler_config.max_num_seqs
        self.device = device
        self.use_cuda_graph = config.suffix_gpu_use_cuda_graph
        self.ingest_chunk = config.suffix_gpu_ingest_chunk
        enable_global = config.suffix_decoding_max_cached_requests != 0

        self.drafter = SuffixGPUDrafter(
            k=self.k,
            device=device,
            max_pattern_len=config.suffix_decoding_max_tree_depth,
            min_match_len=1,
            max_occurrences=config.suffix_gpu_max_occurrences,
            enable_global=enable_global,
            global_capacity=config.suffix_gpu_global_capacity,
            delta_capacity=config.suffix_gpu_delta_capacity,
            rebuild_stream=(
                torch.cuda.Stream(device) if device.type == "cuda" else None
            ),
            max_spec_factor=config.suffix_decoding_max_spec_factor,
            max_spec_offset=0.0,
            min_token_prob=config.suffix_decoding_min_token_prob,
            num_backoff=config.suffix_gpu_num_backoff,
        )

        # Read by the runner's generic spec-decode paths.
        self.draft_logits: torch.Tensor | None = None
        self.supports_mm_inputs = False

        self.req_states: RequestState | None = None
        self._draft_mask = torch.zeros(
            self.max_num_reqs, dtype=torch.bool, device=device
        )
        self._dummy_draft = torch.zeros(
            self.max_num_reqs, self.k, dtype=torch.int64, device=device
        )

        self._warmed_up = False
        self._graph: torch.cuda.CUDAGraph | None = None
        self._graph_outputs: tuple[torch.Tensor, torch.Tensor] | None = None
        self._graph_failed = False

        # Per-slot valid-draft counts, copied D2H on a side stream each
        # step. The runner reports the padded remainder of the previous
        # step's drafts as num_invalid_spec_tokens so metrics do not
        # count padding as real drafts (the V1 runner reports the same
        # via copy_num_valid_draft_tokens).
        self._num_valid_cpu = torch.zeros(
            self.max_num_reqs,
            dtype=torch.int32,
            pin_memory=device.type == "cuda",
        )
        self._num_valid_gpu: torch.Tensor | None = None
        self._copy_stream: torch.cuda.Stream | None = None
        self._copy_event: torch.cuda.Event | None = None
        if device.type == "cuda":
            self._copy_stream = torch.cuda.Stream(device)
            self._copy_event = torch.cuda.Event()

        # Global-index ingestion runs on a side stream so its delta
        # copies stay off the step critical path; sync_pending_ingest()
        # orders later default-stream work after the pending reads.
        self._ingest_stream: torch.cuda.Stream | None = None
        self._ingest_event: torch.cuda.Event | None = None
        self._ingest_pending = False
        if device.type == "cuda" and self.drafter.global_index is not None:
            self._ingest_stream = torch.cuda.Stream(device)
            self._ingest_event = torch.cuda.Event()

    def set_request_states(self, req_states: RequestState) -> None:
        assert req_states.max_num_reqs == self.max_num_reqs
        self.req_states = req_states

    def init_cudagraph_manager(self, cudagraph_mode: CUDAGraphMode) -> None:
        # The drafter captures its own fixed-shape graph; the target
        # model's cudagraph mode does not constrain it.
        pass

    def capture(self) -> None:
        """Warm up (Triton JIT) and capture the fixed-shape draft graph.

        Called by the runner during graph capture; propose() also calls
        it lazily (e.g. enforce_eager engines, where capture never runs).
        Warmup runs even when the CUDA graph is disabled — only the
        capture itself is gated.
        """
        if self.device.type != "cuda" or self.req_states is None:
            return
        if not self._warmed_up:
            self._warmup()
        if not self.use_cuda_graph or self._graph is not None or self._graph_failed:
            return
        try:
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                draft, num_valid = self.drafter.propose(
                    self.req_states.total_len.gpu,
                    self.req_states.all_token_ids.gpu,
                    self._draft_mask,
                )
            self._graph = graph
            self._graph_outputs = (draft, num_valid)
            logger.info_once(
                "suffix_gpu: draft path captured into a CUDA graph (slots=%d, k=%d)",
                self.max_num_reqs,
                self.k,
            )
        except Exception:
            logger.exception(
                "suffix_gpu: CUDA graph capture failed; falling back to eager kernels."
            )
            self._graph_failed = True
            self._graph = None
            self._graph_outputs = None

    def _warmup(self) -> None:
        """JIT-compile the Triton kernels on the resident buffers.

        propose() only reads the token buffer, so warming on the live
        RequestState tensors (with synthetic lengths) is safe.
        """
        assert self.req_states is not None
        counts = torch.randint(
            1,
            max(2, min(64, self.max_model_len)),
            (self.max_num_reqs,),
            dtype=torch.int32,
            device=self.device,
        )
        mask = torch.ones(self.max_num_reqs, dtype=torch.bool, device=self.device)
        for _ in range(3):
            self.drafter.propose(counts, self.req_states.all_token_ids.gpu, mask)
        torch.accelerator.synchronize(self.device)
        self._warmed_up = True

    def propose(
        self,
        input_batch: InputBatch,
        attn_metadata: dict[str, Any],
        slot_mappings: dict[str, torch.Tensor],
        last_hidden_states: torch.Tensor,
        aux_hidden_states: list[torch.Tensor] | None,
        num_sampled: torch.Tensor,
        num_rejected: torch.Tensor,
        last_sampled: torch.Tensor,
        next_prefill_tokens: torch.Tensor,
        temperature: torch.Tensor,
        seeds: torch.Tensor,
        num_tokens_across_dp: torch.Tensor | None = None,
        dummy_run: bool = False,
        skip_attn_for_dummy_run: bool = False,
        mm_inputs: tuple[list[torch.Tensor], torch.Tensor] | None = None,
        is_profile: bool = False,
    ) -> torch.Tensor:
        num_reqs = input_batch.num_reqs
        if dummy_run or is_profile:
            # No collectives on the draft path, so dummy propose does not
            # need to run the kernels for DP/EP sync.
            return self._dummy_draft[:num_reqs]

        assert self.req_states is not None

        # Host-side upkeep: swap in finished background SA rebuilds.
        self.drafter.poll()
        # Chunked global-index ingestion of in-flight responses.
        self._ingest_active(input_batch)
        # Global-index queries below must see the pending ingest.
        self.sync_pending_ingest()

        if not self._warmed_up:
            self._warmup()

        total_len = self.req_states.total_len.gpu
        token_ids = self.req_states.all_token_ids.gpu

        # Rebuild the active-slot mask: draft only for slots in this
        # batch that accepted at least one token (partial prefills
        # sample none) and still have room to grow.
        mask = self._draft_mask
        mask.zero_()
        mask[input_batch.idx_mapping] = num_sampled > 0
        mask &= total_len < self.max_model_len

        if self.use_cuda_graph and self._graph is None and not self._graph_failed:
            # Fallback for engines that never ran capture().
            self.capture()

        if self._graph is not None:
            assert self._graph_outputs is not None
            self._graph.replay()
            draft_full, num_valid_full = self._graph_outputs
        else:
            draft_full, num_valid_full = self.drafter.propose(
                total_len, token_ids, mask
            )
        self._copy_num_valid(num_valid_full)

        draft = draft_full[input_batch.idx_mapping].to(torch.int64)
        # Invalid slots are -1-padded by the drafter. Clamp them to a
        # valid token id: embedding does not tolerate negative ids under
        # TP=1, and a clamped pad is still verified token-by-token by
        # rejection sampling, so correctness is unaffected.
        return draft.clamp_(min=0)

    def _copy_num_valid(self, num_valid_full: torch.Tensor) -> None:
        """Start the D2H copy of this step's per-slot valid counts.

        The runner consumes them one step later, when these drafts are
        verified, so a single buffer suffices: take_invalid_spec_tokens
        always runs before the next propose overwrites it.
        """
        if self._copy_stream is None:
            self._num_valid_cpu.copy_(num_valid_full)
            return
        assert self._copy_event is not None
        self._copy_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self._copy_stream):
            self._num_valid_cpu.copy_(num_valid_full, non_blocking=True)
            self._copy_event.record()
        # Keep the producer tensor alive until the next step so the
        # allocator cannot hand its memory to later default-stream work
        # while the side-stream copy is still reading it.
        self._num_valid_gpu = num_valid_full

    def take_invalid_spec_tokens(
        self, input_batch: InputBatch
    ) -> dict[str, int] | None:
        """Padded (never-valid) slots among this step's scheduled drafts.

        Under async scheduling every running request gets k placeholder
        slots regardless of the drafter's actual emission; report the
        remainder so spec-decode metrics count only real drafts.
        """
        num_draft_tokens_per_req = input_batch.num_draft_tokens_per_req
        if num_draft_tokens_per_req is None:
            return None
        if self._copy_event is not None:
            self._copy_event.synchronize()
        num_valid = self._num_valid_cpu.numpy()
        counts: dict[str, int] = {}
        for i, req_id in enumerate(input_batch.req_ids):
            num_draft_tokens = int(num_draft_tokens_per_req[i])
            if num_draft_tokens <= 0:
                continue
            req_idx = int(input_batch.idx_mapping_np[i])
            invalid = num_draft_tokens - int(num_valid[req_idx])
            if invalid > 0:
                counts[req_id] = invalid
        return counts or None

    # ------------------------------------------------------------------
    # global-memory ingestion (host-side, off the draft path)
    # ------------------------------------------------------------------
    def _ingest_async(
        self,
        keys: list[str],
        rows: list[torch.Tensor],
        lengths: list[int],
        final: bool = False,
    ) -> None:
        if self._ingest_stream is None:
            self.drafter.ingest_active(
                keys, rows, lengths, final=final, chunk=self.ingest_chunk
            )
            return
        # The event is created together with the stream.
        assert self._ingest_event is not None
        default_stream = torch.cuda.current_stream()
        with torch.cuda.stream(self._ingest_stream):
            # Token rows are written on the default stream.
            self._ingest_stream.wait_stream(default_stream)
            self.drafter.ingest_active(
                keys, rows, lengths, final=final, chunk=self.ingest_chunk
            )
            self._ingest_event.record()
        self._ingest_pending = True

    def sync_pending_ingest(self) -> None:
        """Order later default-stream work after pending ingest reads.

        Must run before rewriting ingested token rows (slot reuse after
        request finish) and before querying the global index.
        """
        if self._ingest_pending:
            torch.cuda.current_stream().wait_event(self._ingest_event)
            self._ingest_pending = False

    def _response_span(self, req_idx: int) -> tuple[int, int]:
        """(start, length) of a slot's response tokens, best effort.

        num_computed_tokens_np is an optimistic upper bound under async
        scheduling, so up to k not-yet-verified tokens may be included;
        the V1 proposer ingested with the same optimism.
        """
        assert self.req_states is not None
        start = int(self.req_states.prompt_len.np[req_idx])
        resp_len = int(self.req_states.num_computed_tokens_np[req_idx]) + 1 - start
        return start, max(0, min(resp_len, self.max_model_len - start))

    def _ingest_active(self, input_batch: InputBatch) -> None:
        """Chunked incremental ingestion of in-flight responses."""
        if self.drafter.global_index is None:
            return
        assert self.req_states is not None
        token_ids = self.req_states.all_token_ids.gpu
        keys: list[str] = []
        rows: list[torch.Tensor] = []
        lengths: list[int] = []
        for i, req_id in enumerate(input_batch.req_ids):
            req_idx = int(input_batch.idx_mapping_np[i])
            start, resp_len = self._response_span(req_idx)
            if resp_len < self.ingest_chunk:
                continue
            keys.append(req_id)
            rows.append(token_ids[req_idx, start : start + resp_len])
            lengths.append(resp_len)
        if keys:
            self._ingest_async(keys, rows, lengths)

    def on_requests_finished(self, finished_req_ids: Iterable[str]) -> None:
        """Final-flush finished requests before their slots are reused."""
        if self.drafter.global_index is None or self.req_states is None:
            return
        token_ids = self.req_states.all_token_ids.gpu
        keys: list[str] = []
        rows: list[torch.Tensor] = []
        lengths: list[int] = []
        for req_id in finished_req_ids:
            req_idx = self.req_states.req_id_to_index.get(req_id)
            if req_idx is None:
                self.drafter._ingested.pop(req_id, None)
                continue
            start, resp_len = self._response_span(req_idx)
            keys.append(req_id)
            rows.append(token_ids[req_idx, start : start + resp_len])
            lengths.append(resp_len)
        if keys:
            self._ingest_async(keys, rows, lengths, final=True)
            # Slot rows may be rewritten (apply_write on the default
            # stream) as soon as the slots are reused; order that after
            # the final-flush reads.
            self.sync_pending_ingest()
