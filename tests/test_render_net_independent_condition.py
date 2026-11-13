"""Independent condition and RGB encoders obey the two-stage freeze contract."""

import copy

import pytest
import torch
from torch.nn import functional as F

from diffusion_policy.model.vision.render_net import RenderNet


@pytest.fixture(autouse=True)
def _single_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _model(*, dropout=.3):
    model = RenderNet(
        7, base_channels=4, conditioning_dim=8, wrench_history_steps=8,
        wrench_layers=2, wrench_dropout=.5, embed_dim=16,
        image_self_attn_layers=1, cross_attn_layers=1, decoder_layers=3,
        patch_size=8, num_heads=4, conv_block="bottleneck", dropout=dropout,
        separate_gate_path=True, residual_scale=.5,
        condition_backbone={"base_channels": 2, "embed_dim": 8,
                            "image_self_attn_layers": 0, "cross_attn_layers": 1,
                            "decoder_layers": 0, "mlp_ratio": 2.},
        phase_prediction={"enabled": True, "hidden_dim": 8},
        high_attention_render={"enabled": True, "initial_strength": .4,
                               "threshold": 0., "detach_for_rgb": False},
        spatial_modulation={"enabled": True},
    )
    model.freeze_wrench_encoder()
    return model


def _observations():
    return ({key: .2 + .6 * torch.rand(2, 2, 3, 17, 25)
             for key in ("side", "wrist")}, torch.randn(2, 2, 8, 7))


def _conditions(model, images, wrench):
    _, masks, prediction = model(images, wrench, condition_only=True,
                                 return_aux=True, return_phase=True)
    return {**masks, **prediction}


def test_compact_condition_stage_skips_rgb_and_keeps_heads_on_patch_grid():
    torch.manual_seed(2101)
    model = _model()
    images, wrench = _observations()
    model.set_training_stage("condition_warmup")
    model.set_stage_mode(True)
    rgb_calls, head_shapes = [], []
    rgb_hook = model.backbone.register_forward_hook(lambda *args: rgb_calls.append(1))
    head_hook = model.mask_head.register_forward_pre_hook(
        lambda _, args: head_shapes.append(tuple(args[0].shape[-2:])))
    rgb_before = {key: value.clone() for key, value in model.backbone.state_dict().items()}
    try:
        predictions = _conditions(model, images, wrench)
        assert head_shapes == [(3, 4), (3, 4)]
        assert not rgb_calls
        for key in images:
            assert predictions[key].shape == (2, 2, 1, 17, 25)
        loss = (F.cross_entropy(predictions["phase_logits"].flatten(0, 1),
                                torch.tensor([0, 1, 1, 2]))
                + (predictions["contact_alpha"] - .7).square().mean()
                + sum((predictions[key] - .8).square().mean() for key in images))
        optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
        loss.backward()
        assert any(p.grad is not None and p.grad.abs().sum() > 0
                   for p in model.gate_backbone.stem.parameters())
        assert all(p.grad is None for p in model.backbone.parameters())
        optimizer.step()
        model.initialize_rgb_from_gate()  # Different widths must never copy.
        for key, value in model.backbone.state_dict().items():
            torch.testing.assert_close(value, rgb_before[key], rtol=0, atol=0)
    finally:
        rgb_hook.remove()
        head_hook.remove()


def test_rgb_updates_full_encoder_while_condition_outputs_and_state_stay_bitwise_frozen():
    torch.manual_seed(2102)
    model = _model()
    images, wrench = _observations()
    model.set_training_stage("rgb")
    model.set_stage_mode(True)
    condition_ids = {id(p) for p in model.gate_parameters()}
    rgb_ids = {id(p) for p in model.rgb_parameters()}
    assert condition_ids and rgb_ids and condition_ids.isdisjoint(rgb_ids)
    assert all(not p.requires_grad for p in model.gate_parameters())
    assert all(p.requires_grad for p in model.rgb_parameters())
    frozen = {name: value.detach().clone() for name, value in model.named_parameters()
              if not value.requires_grad}
    before = {key: value.detach().clone()
              for key, value in _conditions(model, images, wrench).items()}
    rgb_stem = model.backbone.stem.body[0].weight.detach().clone()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.01, weight_decay=.01)
    for step in range(3):
        model.set_stage_mode(True)
        optimizer.zero_grad(set_to_none=True)
        rendered, masks, prediction = model(images, wrench, return_aux=True, return_phase=True)
        for key, value in {**masks, **prediction}.items():
            assert not value.requires_grad
            torch.testing.assert_close(value, before[key], rtol=0, atol=0)
        loss = sum((rendered[key] - images[key] - .05).square().mean() for key in images)
        loss.backward()
        if step:
            assert any(p.grad is not None and p.grad.abs().sum() > 0
                       for p in model.backbone.stem.parameters())
        optimizer.step()
    assert not torch.equal(rgb_stem, model.backbone.stem.body[0].weight)
    for name, value in model.named_parameters():
        if name in frozen:
            assert value.grad is None
            torch.testing.assert_close(value, frozen[name], rtol=0, atol=0)
    after = _conditions(model, images, wrench)
    for key in before:
        torch.testing.assert_close(after[key], before[key], rtol=0, atol=0)


def test_independent_condition_predictions_remain_causal():
    torch.manual_seed(2103)
    model = _model().eval()
    images, wrench = _observations()
    before = _conditions(model, images, wrench)
    for image in images.values():
        image[:, 1].normal_(mean=20, std=4)
    wrench[:, 1].normal_(mean=30, std=6)
    after = _conditions(model, images, wrench)
    for key in before:
        torch.testing.assert_close(after[key][:, 0], before[key][:, 0], rtol=0, atol=0)


def test_independent_condition_config_requires_an_independent_path():
    with pytest.raises(ValueError, match="condition_backbone requires"):
        RenderNet(7, condition_backbone={})


def test_dynamic_rgb_compile_reuses_graph_for_changing_contact_counts():
    """Contact filtering must not compile a fresh RGB graph for every count."""
    from torch._dynamo.backends.registry import lookup_backend

    torch.manual_seed(2104)
    eager = _model(dropout=0.)
    eager.set_training_stage("rgb")
    eager.set_stage_mode(True)
    with torch.no_grad():
        # A zero output layer would conceal errors in the encoder's gradients.
        eager.output.weight.normal_(std=.03)
        for modulation in eager.spatial_modulations:
            modulation.affine[-1].weight.normal_(std=.02)
    compiled = copy.deepcopy(eager)
    graphs = []
    aot_eager = lookup_backend("aot_eager")

    def count_graph(graph, example_inputs):
        graphs.append(graph)
        return aot_eager(graph, example_inputs)

    compiled.compile_backbone(backend=count_graph, dynamic=True, fullgraph=True)
    images, wrench = _observations()
    for count in (2, 3, 4, 2):
        selected = (torch.arange(4) < count).reshape(2, 2)
        outputs = []
        for model in (eager, compiled):
            model.zero_grad(set_to_none=True)
            rendered = model(images, wrench, render_observation_mask=selected)
            loss = sum((rendered[key] - images[key] - .05).square().mean()
                       for key in images)
            loss.backward()
            outputs.append(rendered)
        for key in images:
            torch.testing.assert_close(outputs[1][key], outputs[0][key])
            torch.testing.assert_close(outputs[1][key][~selected], images[key][~selected],
                                       rtol=0, atol=0)
        for name, expected in eager.named_parameters():
            actual = compiled.get_parameter(name)
            if expected.grad is None:
                assert actual.grad is None, name
            else:
                torch.testing.assert_close(actual.grad, expected.grad, rtol=1e-4, atol=2e-5,
                                           msg=lambda message: name + ": " + message)
        assert any(p.grad is not None and p.grad.abs().sum() > 0
                   for p in compiled.backbone.stem.parameters())
    # A first graph can specialize batch size 2 to another input dimension.
    # After generalization, new contact counts must reuse that dynamic graph.
    assert len(graphs) <= 2
