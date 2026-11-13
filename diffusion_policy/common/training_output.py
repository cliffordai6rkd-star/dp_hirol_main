"""Reset one selected training directory after Hydra has initialized its files."""

import logging
from pathlib import Path
import shutil


def reset_training_output(output_dir, *, hydra_output_subdir=".hydra", protected_paths=()):
    output = Path(output_dir).expanduser().resolve()
    for protected in protected_paths:
        if Path(protected).expanduser().resolve().is_relative_to(output):
            raise ValueError(f"Refusing to overwrite a directory containing project files: {output}")
    output.mkdir(parents=True, exist_ok=True)

    # Hydra has already written this invocation's config and opened its job
    # log. Keep those new configs and reset open logs in place, so subsequent
    # messages do not go to unlinked files.
    preserved = set()
    if hydra_output_subdir is not None:
        preserved.add(output / hydra_output_subdir)
    loggers = [logging.getLogger()] + [
        logger for logger in list(logging.Logger.manager.loggerDict.values())
        if isinstance(logger, logging.Logger)
    ]
    handlers = {
        handler for logger in loggers for handler in logger.handlers
        if isinstance(handler, logging.FileHandler)
    }
    for handler in handlers:
        path = Path(handler.baseFilename).resolve()
        if not path.is_relative_to(output):
            continue
        preserved.add(path)
        handler.acquire()
        try:
            handler.flush()
            if handler.stream is not None:
                handler.stream.seek(0)
                handler.stream.truncate(0)
            else:
                path.write_text("")
        finally:
            handler.release()

    def clear(directory):
        for path in directory.iterdir():
            if path in preserved:
                continue
            if path.is_dir() and not path.is_symlink():
                if any(path in kept.parents for kept in preserved):
                    clear(path)
                else:
                    shutil.rmtree(path)
            else:
                path.unlink()

    clear(output)
