"""Final-context token alignment weighted by a frozen teacher's force response."""

import torch


def response_weighted_token_mse(student, teacher, response, *, noise_floor=1e-6):
    """Return global MSE, response MSE and its unnormalized numerator/weight sum.

    Responses have the same [batch, token] order as the context. Normalize over
    the complete batch, including all observation times and cameras. Neither
    the teacher target nor the response receives gradients.
    """
    if student.shape != teacher.shape or response.shape != student.shape[:2]:
        raise ValueError("Token responses must match the [B, N] context layout")
    errors = (student.float() - teacher.detach().float()).square().mean(dim=-1)
    weights = (response.detach().float() - noise_floor).clamp_min(0)
    numerator = (weights * errors).sum()
    denominator = weights.sum()
    weighted = numerator / denominator.clamp_min(torch.finfo(torch.float32).eps)
    return errors.mean(), weighted, numerator, denominator
