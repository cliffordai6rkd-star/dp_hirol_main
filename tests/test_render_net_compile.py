"""Compilation follows the shapes used by sparse RGB rendering."""

from types import SimpleNamespace

from omegaconf import OmegaConf
import pytest
import torch

from diffusion_policy.model.vision.render_net import RenderNet
from diffusion_policy.workspace.train_distill_render_net_workspace import TrainDistillRenderNetWorkspace
