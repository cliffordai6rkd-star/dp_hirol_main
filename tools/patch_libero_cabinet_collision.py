#!/usr/bin/env python3
"""Add a small safety margin to LIBERO wooden-cabinet collision proxies.

The cabinet visual meshes are intentionally non-colliding in LIBERO.  Collision
is provided by the group=0 box proxies.  A 2 mm MuJoCo margin makes those
proxies cover small visual/proxy gaps and lets contact be detected earlier.
"""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path


GEOM_RE = re.compile(r"(<geom\b)([^>]*\btype=\"box\")(.*)>")


def patch_file(path: Path, margin: str, backup: bool) -> int:
    original = path.read_text()
    if backup:
        backup_path = path.with_suffix(path.suffix + ".pre_collision_margin")
        if not backup_path.exists():
            shutil.copy2(path, backup_path)

    changed = 0

    def replace(match: re.Match[str]) -> str:
        nonlocal changed
        head, attrs, tail = match.groups()
        full = head + attrs + tail
        if "margin=" in full:
            return match.group(0)
        changed += 1
        return f'{head} margin="{margin}"{attrs}{tail}>'

    updated = GEOM_RE.sub(replace, original)
    if changed:
        path.write_text(updated)
    return changed


def cabinet_files(root: Path) -> list[Path]:
    directory = root / "libero/libero/assets/articulated_objects"
    files = [directory / "wooden_cabinet.xml"]
    files.extend(sorted((directory / "wooden_cabinet").glob("**/*.xml")))
    return files


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", action="append", required=True, type=Path)
    parser.add_argument("--margin", default="0.002")
    parser.add_argument("--no-backup", action="store_true")
    parser.add_argument("--restore", action="store_true")
    args = parser.parse_args()

    total = 0
    for root in args.root:
        for path in cabinet_files(root):
            if not path.exists():
                raise FileNotFoundError(path)
            if args.restore:
                backup_path = path.with_suffix(path.suffix + ".pre_collision_margin")
                if not backup_path.exists():
                    raise FileNotFoundError(backup_path)
                shutil.copy2(backup_path, path)
                print(f"{path}: restored from {backup_path}")
                continue
            changed = patch_file(path, args.margin, backup=not args.no_backup)
            total += changed
            print(f"{path}: changed {changed} box geoms")
    print(f"total changed: {total}")


if __name__ == "__main__":
    main()
