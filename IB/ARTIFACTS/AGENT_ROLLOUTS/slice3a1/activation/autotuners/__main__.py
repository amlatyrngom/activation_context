"""Inspect shared results, retune model geometries, or promote validated JSON."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

from .core import CONFIG_ROOT, publish, read_profile, resolve
from ..common.data_syncing import resolve_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("inspect")
    promote = commands.add_parser("promote")
    promote.add_argument("profile", type=Path)
    retune = commands.add_parser("retune")
    retune.add_argument("model_id", nargs="+")
    retune.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.command == "inspect":
        for folder in (resolve_path("AUTOTUNE/profiles"), CONFIG_ROOT):
            for path in sorted(folder.glob("*.json")):
                profile = read_profile(path)
                print(json.dumps({"path": str(path), "valid": profile is not None,
                                  "identity": profile["identity"] if profile else None,
                                  "tuning": profile["tuning"] if profile else None}))
    elif args.command == "promote":
        profile = read_profile(args.profile)
        if profile is None:
            parser.error("Only complete, checksummed validated profiles can be promoted")
        print(publish(profile, CONFIG_ROOT))
    else:
        import torch
        from transformers import AutoConfig
        from ..harness.model_config import ModelConfig
        for model_id in args.model_id:
            AutoConfig.from_pretrained(model_id)  # explicit CLI metadata preparation, before the tuning budget
        models = [ModelConfig(model_id, model_id) for model_id in args.model_id]
        config = SimpleNamespace(logits_chunk_tokens=None, gradient_checkpointing_min_tokens=None)
        values, _ = resolve(models, config, torch.device(args.device), None, retune=True)
        print(json.dumps(values["provenance"]))


if __name__ == "__main__":
    main()
