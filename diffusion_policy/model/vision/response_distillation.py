"""Final-context token alignment weighted by a frozen teacher's force response."""

import torch


def phase_token_mask(tokens, phases):
    """Expand offline precontact/contact labels to [B, observation*patches].

    This supervision mask never uses the student's predicted rendering gate.
    Missing phase labels retain the legacy, phase-unaware caller behavior.
    """
    if phases is None:
        return torch.ones(tokens.shape[:2], device=tokens.device, dtype=torch.bool)
    if phases.ndim != 2 or phases.shape[0] != tokens.shape[0] or phases.shape[1] < 1:
        raise ValueError("Phase labels must have shape [B, observation_steps]")
    steps = phases.shape[1]
    if tokens.shape[1] % steps:
        raise ValueError("Context token layout must be divisible by observation steps")
    selected = ((phases == 1) | (phases == 2)).to(device=tokens.device)
    return selected.unsqueeze(-1).expand(-1, -1, tokens.shape[1] // steps).reshape(tokens.shape[:2])


def masked_token_mse(student, teacher, token_mask=None):
    """Return selected-token MSE, additive error sum and selected-token count."""
    if student.shape != teacher.shape or student.ndim != 3:
        raise ValueError("Student and teacher contexts must share [B, N, D] shape")
    errors = (student.float() - teacher.detach().float()).square().mean(dim=-1)
    if token_mask is None:
        token_mask = torch.ones_like(errors, dtype=torch.bool)
    if token_mask.shape != errors.shape:
        raise ValueError("Token mask must match the [B, N] context layout")
    numerator = torch.where(token_mask, errors, 0).sum()
    count = token_mask.sum()
    return numerator / count.clamp_min(1), numerator, count


def response_weighted_token_mse(student, teacher, response, *, noise_floor=1e-6,
                                token_mask=None):
    """Return selected MSE, response MSE and its unnormalized numerator/weight sum.

    Responses have the same [batch, token] order as the context. Normalize over
    selected tokens in the batch, including all selected observation times and
    cameras. Neither the teacher target nor the response receives gradients.
    """
    if student.shape != teacher.shape or response.shape != student.shape[:2]:
        raise ValueError("Token responses must match the [B, N] context layout")
    errors = (student.float() - teacher.detach().float()).square().mean(dim=-1)
    weights = (response.detach().float() - noise_floor).clamp_min(0)
    if token_mask is not None:
        weights = torch.where(token_mask, weights, 0)
    numerator = (weights * errors).sum()
    denominator = weights.sum()
    weighted = numerator / denominator.clamp_min(torch.finfo(torch.float32).eps)
    token_mse, _, _ = masked_token_mse(student, teacher, token_mask)
    return token_mse, weighted, numerator, denominator
