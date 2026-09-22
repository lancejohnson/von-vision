"""Small local command-line entry point for Von Vision."""

from __future__ import annotations

import argparse
import json

from .scorer import VisionScorer


def main() -> None:
    parser = argparse.ArgumentParser(description="Score local images with SigLIP/SigLIP2.")
    parser.add_argument("image", nargs="+", help="Local image path(s)")
    parser.add_argument("--option", action="append", required=True, metavar="KEY=TEXT", help="Candidate option")
    parser.add_argument("--instructions", default="", help="Question applied to each option")
    parser.add_argument("--aggregation", choices=["mean", "max"], default="mean")
    parser.add_argument("--model", default="google/siglip2-base-patch16-224")
    args = parser.parse_args()
    criteria = {}
    for item in args.option:
        if "=" not in item:
            parser.error("--option must use KEY=TEXT")
        key, text = item.split("=", 1)
        criteria[key] = text
    result = VisionScorer(args.model, aggregation=args.aggregation).choice(
        args.image, criteria, instructions=args.instructions
    )
    print(json.dumps(result.__dict__, indent=2))


if __name__ == "__main__":
    main()
