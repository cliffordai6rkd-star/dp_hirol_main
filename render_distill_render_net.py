#!/usr/bin/env python3
"""Run RGB rendering inference for the current-wrench RenderNet checkpoint.

This is a small user-facing entry point around
``diffusion_policy.scripts.visualize_render_net_episode``.  The underlying
renderer restores the EMA RenderNet weights and the wrench preprocessing
state saved in the checkpoint, then writes both a rendered-RGB video and a
comparison video for the selected dataset episode.

Examples
--------
Render episode 0 with the checkpoint from this training run::

    conda activate fdp
    python render_distill_render_net.py --episode 0 --overwrite

Use another episode or render only a short prefix while checking the setup::

    python render_distill_render_net.py --episode 3 --max-frames 100
"""

from __future__ import annotations

import pathlib
import sys


REPO_ROOT = pathlib.Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = (
    REPO_ROOT
    / "data"
    / "outputs"
    / "civi"
    / "current_wrench"
    / "student"
    / "checkpoints"
    / "latest.ckpt"
)


def _has_option(argv: list[str], option: str) -> bool:
    """Return whether *argv* contains an argparse long option."""
    return any(arg == option or arg.startswith(f"{option}=") for arg in argv)


def main(argv: list[str] | None = None) -> None:
    # Import after resolving the repository root so the script also works when
    # launched from a different working directory.
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

    from diffusion_policy.scripts.visualize_render_net_episode import (
        parse_args,
        run,
    )

    forwarded = list(sys.argv[1:] if argv is None else argv)
    if not _has_option(forwarded, "--checkpoint"):
        forwarded = ["--checkpoint", str(DEFAULT_CHECKPOINT), *forwarded]
    if not _has_option(forwarded, "--episode"):
        # Episode 0 makes a no-argument render invocation useful while still
        # allowing callers to select any episode explicitly.
        forwarded = ["--episode", "0", *forwarded]
    # The defaults are injected above; make argparse present them as optional
    # in this convenience entry point's help text as well.
    run(parse_args(forwarded, checkpoint_required=False, episode_required=False))


if __name__ == "__main__":
    main()
