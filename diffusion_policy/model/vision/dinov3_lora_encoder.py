"""An independent pretrained DINOv3 visual encoder with native PyTorch LoRA."""

import math

import torch
from torch import nn


class LoRALinear(nn.Module):
    """Keep a pretrained projection frozen and learn a zero-initialized update."""

    def __init__(self, base, *, rank, alpha, dropout):
        super().__init__()
        self.base = base.requires_grad_(False)
        self.in_features = base.in_features
        self.out_features = base.out_features
        self.scaling = float(alpha) / rank
        factory = {"device": base.weight.device, "dtype": base.weight.dtype}
        self.lora_A = nn.Linear(base.in_features, rank, bias=False, **factory)
        self.lora_B = nn.Linear(rank, base.out_features, bias=False, **factory)
        self.dropout = nn.Dropout(dropout)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, value):
        update = self.lora_B(self.lora_A(self.dropout(value)))
        return self.base(value) + self.scaling * update

    def train(self, mode=True):
        super().train(mode)
        self.base.eval()
        return self

    def lora_parameters(self):
        yield from self.lora_A.parameters()
        yield from self.lora_B.parameters()


class DINOv3LoRAEncoder(nn.Module):
    """Encode RGB with a private DINOv3 copy and trainable attention adapters.

    ``intermediate_layers`` contains zero-based transformer layer indices.
    The final and intermediate outputs all use the pretrained final layer
    norm and omit CLS/register tokens. For a twelve-layer model the default
    intermediate layers are 2, 5 and 8.
    """

    def __init__(self, pretrained_model_name_or_path, *, rank=8, alpha=16.0,
                 dropout=0.0, target_modules=("q_proj", "v_proj"),
                 local_files_only=True, intermediate_layers=None):
        super().__init__()
        if isinstance(rank, bool) or not isinstance(rank, int) or rank < 1:
            raise ValueError("LoRA rank must be a positive integer")
        if not math.isfinite(alpha) or alpha <= 0:
            raise ValueError("LoRA alpha must be positive and finite")
        if not math.isfinite(dropout) or not 0 <= dropout < 1:
            raise ValueError("LoRA dropout must lie in [0, 1)")
        targets = tuple(target_modules)
        if not targets or any(not isinstance(name, str) or not name for name in targets):
            raise ValueError("target_modules must contain projection names")

        from transformers import AutoModel

        # Always load a new model: adapters must never modify the teacher.
        self.backbone = AutoModel.from_pretrained(
            pretrained_model_name_or_path, local_files_only=local_files_only,
        )
        config = self.backbone.config
        if (getattr(config, "model_type", None) != "dinov3_vit"
                or not all(hasattr(self.backbone, name)
                           for name in ("embeddings", "rope_embeddings", "layer", "norm"))):
            raise ValueError("DINOv3LoRAEncoder requires a DINOv3 ViT backbone")
        self.hidden_size = int(config.hidden_size)
        self.patch_size = config.patch_size
        self.num_register_tokens = int(config.num_register_tokens)
        self.pretrained_model_name_or_path = str(pretrained_model_name_or_path)
        depth = len(self.backbone.layer)
        if intermediate_layers is None:
            intermediate_layers = sorted({max(0, depth * fraction // 4 - 1)
                                          for fraction in (1, 2, 3)})
        self.intermediate_layers = tuple(intermediate_layers)
        if (any(isinstance(index, bool) or not isinstance(index, int)
                or not 0 <= index < depth for index in self.intermediate_layers)
                or tuple(sorted(set(self.intermediate_layers))) != self.intermediate_layers):
            raise ValueError("intermediate_layers must be increasing unique indices within the DINO depth")

        self.backbone.requires_grad_(False)
        adapted = []
        matched = set()
        for full_name, module in tuple(self.backbone.named_modules()):
            parent_name, _, name = full_name.rpartition(".")
            if name not in targets or not isinstance(module, nn.Linear):
                continue
            adapter = LoRALinear(module, rank=rank, alpha=alpha, dropout=dropout)
            setattr(self.backbone.get_submodule(parent_name), name, adapter)
            adapted.append(adapter)
            matched.add(name)
        if set(targets) != matched:
            raise ValueError(f"DINO attention projections were not found: {sorted(set(targets) - matched)}")
        # The modules are registered under backbone; this tuple only provides
        # a direct view for stage ownership and dropout mode changes.
        self._lora_layers = tuple(adapted)
        self.register_buffer("image_mean", torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1),
                             persistent=False)
        self.register_buffer("image_std", torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1),
                             persistent=False)
        self.train()

    def lora_parameters(self):
        for adapter in self._lora_layers:
            yield from adapter.lora_parameters()

    def enforce_base_freeze(self):
        """Restore base freezing without reopening adapters frozen by a stage."""
        adapter_ids = {id(parameter) for parameter in self.lora_parameters()}
        for parameter in self.backbone.parameters():
            if id(parameter) not in adapter_ids:
                parameter.requires_grad_(False)

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        for adapter in self._lora_layers:
            adapter.train(mode)
        return self

    def forward(self, image):
        if image.ndim != 4 or image.shape[1] != 3:
            raise ValueError("DINOv3LoRAEncoder expects RGB [B, 3, H, W] in [0, 1]")
        image = (image - self.image_mean.to(image)) / self.image_std.to(image)
        outputs = self.backbone(pixel_values=image, output_hidden_states=True, return_dict=True)
        patch_start = 1 + self.num_register_tokens
        # HF records embeddings at index zero and each transformer output at
        # index layer+1; those intermediate states precede the final norm.
        intermediate = tuple(self.backbone.norm(outputs.hidden_states[index + 1])[:, patch_start:]
                             for index in self.intermediate_layers)
        return outputs.last_hidden_state[:, patch_start:], intermediate
