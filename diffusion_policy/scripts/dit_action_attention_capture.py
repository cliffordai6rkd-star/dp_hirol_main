"""Capture DiT action-to-context attention paths for one denoising call.

The result is an equal-residual attention rollout diagnostic.  Each denoiser
call is retained separately; it is not an attribution of the full diffusion
sampler or final action.
"""

import inspect
import operator

import torch
import torch.nn as nn

from diffusion_policy.model.diffusion.context_transformer_for_diffusion import (
    ContextTransformerForDiffusion,
)


class DitActionAttentionCapture:
    """Temporarily capture decoder action reads back to input context tokens.

    ``layer_index`` selects a decoder prefix ending at that layer. ``executed``
    (the default action query) selects the policy's executed action chunk;
    ``first``, ``last`` and ``all`` select positions in the full horizon.
    ``action_index`` overrides this with one absolute horizon index. The
    ``denoising_step`` selector chooses one sampling call, defaulting to the
    final call. Use :meth:`step_context_relevance` to inspect every call.
    """

    def __init__(self, policy, layer_index=-1, action_query="executed",
                 action_index=None, denoising_step="last"):
        self.policy = policy
        self.model = getattr(policy, "model", None)
        if not isinstance(self.model, ContextTransformerForDiffusion):
            raise ValueError("DiT rollout requires ContextTransformerForDiffusion")
        self.decoder_layers = tuple(self.model.decoder.layers)
        if not self.decoder_layers:
            raise ValueError("DiT rollout requires at least one decoder layer")
        self.layer_index = self._resolve_index(layer_index, len(self.decoder_layers), "layer_index")
        condition = self.model.condition_encoder
        if isinstance(condition, nn.Identity):
            self.condition_layers = ()
        elif isinstance(condition, nn.TransformerEncoder):
            self.condition_layers = tuple(condition.layers)
        else:
            raise ValueError("Unsupported DiT condition_encoder")
        for layer in self.condition_layers:
            if not isinstance(layer, nn.TransformerEncoderLayer) or not layer.norm_first:
                raise ValueError("DiT condition encoder requires pre-norm TransformerEncoderLayer")
        for layer in self.decoder_layers:
            if not isinstance(layer, nn.TransformerDecoderLayer) or not layer.norm_first:
                raise ValueError("DiT decoder requires pre-norm TransformerDecoderLayer")
        if action_query not in {"executed", "all", "first", "last"}:
            raise ValueError("action_query must be executed, all, first or last")
        self.action_query = action_query
        self.action_index = None if action_index is None else self._integer(action_index, "action_index")
        if isinstance(denoising_step, str) and denoising_step in {"first", "last"}:
            self.denoising_step = denoising_step
        else:
            try:
                self.denoising_step = int(denoising_step) if isinstance(denoising_step, str) else self._integer(denoising_step, "denoising_step")
            except (TypeError, ValueError) as exc:
                raise ValueError("denoising_step must be first, last or a single sampling-call index") from exc
        self.handles = []
        self.active = False
        self._clear()

    @staticmethod
    def _integer(value, name):
        try:
            return operator.index(value)
        except TypeError as exc:
            raise ValueError(f"{name} must be an integer") from exc

    @classmethod
    def _resolve_index(cls, index, count, name):
        index = cls._integer(index, name)
        if not -count <= index < count:
            raise ValueError(f"{name} must be between {-count} and {count - 1}")
        return index % count

    def _clear(self):
        self._steps = []
        self._current = None

    def __enter__(self):
        if self.active:
            raise RuntimeError("DiT attention capture is already active")
        if self.model.training:
            raise ValueError("DiT attention visualization requires an eval-mode model")
        self.active = True
        self._clear()
        try:
            encoder = getattr(self.policy, "obs_encoder", None)
            if isinstance(encoder, nn.Module):
                self.handles.append(encoder.register_forward_pre_hook(self._reset_prediction))
            self.handles.append(self.model.register_forward_pre_hook(self._begin_step, with_kwargs=True))
            self.handles.append(self.model.register_forward_hook(self._finish_step))
            for index, layer in enumerate(self.condition_layers):
                self._attach(layer.self_attn, "condition", index)
            for index, layer in enumerate(self.decoder_layers[:self.layer_index + 1]):
                self._attach(layer.self_attn, "self", index)
                self._attach(layer.multihead_attn, "cross", index)
        except BaseException:
            self.__exit__(Exception, None, None)
            raise
        return self

    def __exit__(self, exc_type, *exc_info):
        for handle in reversed(self.handles):
            handle.remove()
        self.handles.clear()
        self.active = False
        if exc_type is not None:
            self._clear()

    def _reset_prediction(self, module, args):
        self._clear()

    def _attach(self, attention, kind, index):
        if not isinstance(attention, nn.MultiheadAttention):
            raise ValueError("DiT attention must use nn.MultiheadAttention")
        self.handles.append(attention.register_forward_pre_hook(self._request_weights, with_kwargs=True))
        self.handles.append(attention.register_forward_hook(
            lambda module, args, output: self._record(kind, index, output)
        ))

    @staticmethod
    def _request_weights(module, args, kwargs):
        args, kwargs = list(args), dict(kwargs)
        for position, name, value in ((4, "need_weights", True), (6, "average_attn_weights", False)):
            if len(args) > position:
                args[position] = value
                kwargs.pop(name, None)
            else:
                kwargs[name] = value
        return tuple(args), kwargs

    def _begin_step(self, module, args, kwargs):
        if self._current is not None:
            raise RuntimeError("Previous DiT forward did not finish")
        inputs = inspect.signature(module.forward).bind(*args, **kwargs).arguments
        sample, context, timestep = (inputs[key] for key in ("sample", "context", "timestep"))
        if sample.ndim != 3 or context.ndim != 3 or sample.shape[0] != 1 or context.shape[0] != 1:
            raise ValueError("DiT episode visualization requires one sample with [1,H,D] actions and [1,S,D] context")
        if context.shape[1] < 1:
            raise ValueError("DiT rollout requires nonempty context")
        if torch.is_tensor(timestep):
            if timestep.numel() != 1:
                raise ValueError("DiT visualization requires one diffusion timestep")
            timestep = timestep.detach().item()
        self._current = {"horizon": sample.shape[1], "context_count": context.shape[1],
                         "timestep": timestep, "condition": {}, "self": {}, "cross": {}}

    def _record(self, kind, index, output):
        if self._current is None:
            raise RuntimeError("Attention captured outside a DiT forward")
        weights = output[1]
        if weights is None or weights.ndim != 4 or weights.shape[0] != 1:
            raise RuntimeError("Unexpected DiT attention shape; expected [1,heads,queries,keys]")
        if index in self._current[kind]:
            raise RuntimeError("Unexpected repeated DiT attention layer")
        self._current[kind][index] = weights.detach().float().mean(dim=1)[0]

    def _action_indices(self, horizon):
        if self.action_index is not None:
            return [self._resolve_index(self.action_index, horizon, "action_index")]
        if self.action_query == "all":
            return list(range(horizon))
        if self.action_query == "first":
            return [0]
        if self.action_query == "last":
            return [horizon - 1]
        start = int(getattr(self.policy, "action_start_index", self.policy.n_obs_steps - 1))
        end = start + int(self.policy.n_action_steps)
        if not 0 <= start < end <= horizon:
            raise ValueError("Executed action chunk does not fit the DiT action horizon")
        return list(range(start, end))

    @staticmethod
    def _residual_attention(weights):
        if weights.shape[0] != weights.shape[1]:
            raise RuntimeError("Self-attention matrix must be square")
        residual = weights + torch.eye(weights.shape[0], device=weights.device, dtype=weights.dtype)
        return residual / residual.sum(dim=-1, keepdim=True)

    @torch.no_grad()
    def _finish_step(self, module, args, output):
        current = self._current
        if current is None:
            raise RuntimeError("DiT forward was not initialized")
        if (len(current["condition"]) != len(self.condition_layers)
                or len(current["self"]) != self.layer_index + 1
                or len(current["cross"]) != self.layer_index + 1):
            raise RuntimeError("Incomplete DiT attention forward pass")
        horizon, memory_count = current["horizon"], current["context_count"] + 1
        reference = current["cross"][0]
        query = reference.new_zeros(horizon)
        indices = self._action_indices(horizon)
        query[indices] = 1.0 / len(indices)
        memory = reference.new_zeros(memory_count)
        for index in range(self.layer_index, -1, -1):
            cross, self_attention = current["cross"][index], current["self"][index]
            if cross.shape != (horizon, memory_count) or self_attention.shape != (horizon, horizon):
                raise RuntimeError("DiT attention token counts do not match actions/context")
            memory += 0.5 * (query @ cross)
            query = (0.5 * query) @ self._residual_attention(self_attention)
        for index in range(len(self.condition_layers) - 1, -1, -1):
            condition = current["condition"][index]
            if condition.shape != (memory_count, memory_count):
                raise RuntimeError("Condition attention token count does not match context")
            memory = memory @ self._residual_attention(condition)
        self._steps.append({"context": memory[1:].cpu(), "time_mass": float(memory[0]),
                            "input_action_mass": float(query.sum()),
                            "timestep": current["timestep"], "action_indices": indices})
        self._current = None

    def _require_complete(self):
        if not self._steps or self._current is not None:
            raise RuntimeError("No complete DiT inference attention was captured")
        if len({step["context"].shape for step in self._steps}) != 1:
            raise RuntimeError("Context token count changed between denoising calls")

    def _selected_steps(self):
        self._require_complete()
        count = len(self._steps)
        if self.denoising_step == "first":
            return [0]
        if self.denoising_step == "last":
            return [count - 1]
        return [self._resolve_index(self.denoising_step, count, "denoising_step")]

    def step_context_relevance(self):
        """Separate rollout for every denoising call, as ``[N,S]``."""
        self._require_complete()
        return torch.stack([step["context"] for step in self._steps])

    def context_relevance(self):
        """One selected denoising call's context rollout, as ``[S]``."""
        return self.step_context_relevance()[self._selected_steps()[0]]

    def diagnostics(self):
        selected = self._selected_steps()
        index = selected[0]
        contexts = self.step_context_relevance()
        time_mass = [step["time_mass"] for step in self._steps]
        action_mass = [step["input_action_mass"] for step in self._steps]
        return {
            "method": "equal_residual_attention_rollout",
            "interpretation": "Separate per-denoiser-call attention paths; no scheduler propagation or final-action attribution.",
            "residual_branch_weight": 0.5,
            "decoder_layer_count": len(self.decoder_layers),
            "selected_decoder_layer": self.layer_index,
            "condition_encoder_layer_count": len(self.condition_layers),
            "action_query": "index" if self.action_index is not None else self.action_query,
            "selected_action_indices": self._steps[0]["action_indices"],
            "denoising_step": self.denoising_step,
            "selected_denoising_indices": selected,
            "timesteps": [step["timestep"] for step in self._steps],
            "selected_denoising_timesteps": [self._steps[index]["timestep"] for index in selected],
            "per_step_time_mass": time_mass,
            "per_step_input_action_mass": action_mass,
            "per_step_context_mass": contexts.sum(dim=1).tolist(),
            "time_mass": time_mass[index],
            "input_action_mass": action_mass[index],
            "context_mass": float(contexts[index].sum()),
        }
