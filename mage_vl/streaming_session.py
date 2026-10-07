"""Incremental StreamMind gate state for Mage-VL live streams.

The released checkpoint exposes a causal gate over a *list* of preprocessed
segments, but that API concatenates the whole stream.  This module keeps the Mamba
inference cache between calls so one completed segment can be pushed at a time with
constant state memory.

This adapter intentionally targets the pinned ``microsoft/Mage-VL`` remote-code API.
It can be removed once the checkpoint exposes an equivalent public step method.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class GateStep:
    """Gate output at the end of one semantic stream segment."""

    logits: Any
    probability: float
    epfe_steps: int
    total_epfe_steps: int
    state_bytes: int


def _tensor_bytes(value: Any) -> int:
    """Recursively count tensor storage in a nested cache structure."""
    try:
        import torch
    except ImportError:
        return 0
    if isinstance(value, torch.Tensor):
        return value.numel() * value.element_size()
    if isinstance(value, dict):
        return sum(_tensor_bytes(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_tensor_bytes(item) for item in value)
    return 0


class IncrementalStreamMindSession:
    """Stateful, constant-memory wrapper around Mage-VL's StreamMind gate.

    Args:
        model: Loaded ``MageVLForConditionalGeneration`` instance.
        max_stream_steps: Safety ceiling for EPFE time steps in one session.  Mamba's
            recurrent cache is constant-size; this bound catches accidental reuse of
            a session across unrelated streams.

    Notes:
        Mamba's offset-zero path prefills the first segment as one causal scan. Once
        the cache offset is nonzero, its cached path accepts one time token per call,
        so later segments are stepped token-by-token. Old visual tensors are never
        retained.
    """

    def __init__(
        self,
        model: Any,
        max_stream_steps: int = 1_000_000,
        boundary_token_id: int = 1,
    ):
        import torch
        from mamba_ssm.utils.generation import InferenceParams

        if max_stream_steps <= 0:
            raise ValueError("max_stream_steps must be positive")
        if boundary_token_id not in (0, 1):
            raise ValueError("boundary_token_id must be 0 or 1")
        core = getattr(model, "model", model)
        required = ("_streammind_vision_tokens", "_load_streammind_gate")
        missing = [name for name in required if not hasattr(core, name)]
        if missing:
            raise TypeError(
                "checkpoint does not expose the pinned StreamMind API: "
                + ", ".join(missing)
            )

        self.model = model
        self.core = core
        self.gate = core._load_streammind_gate()
        self.max_stream_steps = int(max_stream_steps)
        self.boundary_token_id = int(boundary_token_id)
        self._InferenceParams = InferenceParams
        self._torch = torch
        self._params = self._new_params()
        self._total_steps = 0

    def _new_params(self):
        return self._InferenceParams(
            max_seqlen=self.max_stream_steps,
            max_batch_size=1,
        )

    @property
    def total_steps(self) -> int:
        return self._total_steps

    @property
    def state_bytes(self) -> int:
        return _tensor_bytes(self._params.key_value_memory_dict)

    def reset(self) -> None:
        """Start a new independent stream and discard all recurrent state."""
        self._params = self._new_params()
        self._total_steps = 0

    def _classify_last_token(self, token):
        """Apply the checkpoint's independent silent/speak classifier."""
        torch = self._torch
        gate = self.gate
        batch, time, dim = token.shape
        if batch != 1 or time != 1:
            raise ValueError(f"expected one [1,1,D] token, got {tuple(token.shape)}")

        target_ids = torch.full(
            (batch, time),
            self.boundary_token_id,
            dtype=torch.long,
            device=token.device,
        )
        target = gate.cls_net.cls_model.model.embed_tokens(target_ids.reshape(-1))
        pair = torch.stack((token.reshape(-1, dim), target), dim=1)
        rotary = gate.cls_net.cls_model.model.rotary_emb
        saved_inv_freq = rotary.inv_freq
        try:
            rotary.inv_freq = rotary.inv_freq.to(pair.dtype)
            output = gate.cls_net(
                pair,
                attention_mask=torch.ones(pair.shape[:2], device=pair.device),
            )
        finally:
            rotary.inv_freq = saved_inv_freq
        return output["logits"][:, 0].reshape(batch, time, 2)[:, -1]

    def push_vision_tokens(self, vision_tokens) -> GateStep:
        """Advance the gate with one segment's ``[1,T,P,D]`` vision tokens."""
        torch = self._torch
        if vision_tokens.ndim != 4 or vision_tokens.shape[0] != 1:
            raise ValueError(
                "vision_tokens must have shape [1,T,P,D], got "
                f"{tuple(vision_tokens.shape)}"
            )
        time = int(vision_tokens.shape[1])
        if time <= 0:
            raise ValueError("segment contains no EPFE time steps")
        if self._total_steps + time > self.max_stream_steps:
            raise RuntimeError(
                f"stream exceeds max_stream_steps={self.max_stream_steps}; "
                "reset the session or increase the limit"
            )

        pooled = vision_tokens.mean(dim=2)
        _, _, dim = pooled.shape
        projected = self.gate.pre_net(pooled.reshape(time, dim)).reshape(1, time, dim)

        if self._total_steps == 0:
            prefill = self.gate.mamba_model(
                projected,
                inference_params=self._params,
            )
            last = prefill[:, -1:]
            self._params.seqlen_offset += time
            self._total_steps += time
        else:
            last = None
            for index in range(time):
                last = self.gate.mamba_model(
                    projected[:, index:index + 1],
                    inference_params=self._params,
                )
                self._params.seqlen_offset += 1
                self._total_steps += 1

        assert last is not None
        token = self.gate.post_net(last.reshape(1, dim)).reshape(1, 1, dim)
        logits = self._classify_last_token(token)
        probability = float(torch.softmax(logits.float(), dim=-1)[0, 1].item())
        return GateStep(
            logits=logits,
            probability=probability,
            epfe_steps=time,
            total_epfe_steps=self._total_steps,
            state_bytes=self.state_bytes,
        )

    @property
    def inference_params(self):
        """Expose state for diagnostics; callers must not mutate it."""
        return self._params

    def push_segment(self, segment_inputs: dict[str, Any]) -> GateStep:
        """Encode and advance one processor output without retaining its tensors."""
        torch = self._torch
        required = ("pixel_values", "image_grid_thw")
        missing = [key for key in required if key not in segment_inputs]
        if missing:
            raise KeyError("segment inputs missing: " + ", ".join(missing))
        with torch.inference_mode():
            vision_tokens = self.core._streammind_vision_tokens(
                segment_inputs["pixel_values"],
                segment_inputs["image_grid_thw"],
                patch_positions=segment_inputs.get("patch_positions"),
            )
            return self.push_vision_tokens(vision_tokens)
