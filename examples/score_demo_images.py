"""Run Von Vision's default SigLIP2 model over the Wikimedia demo set."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from von_vision import VisionScorer


VIEW_TYPES = {
    "water": "A property photograph dominated by an expansive ocean, lake, or other water view.",
    "mountains": "A property photograph dominated by an expansive mountain view.",
    "city": "A property photograph dominated by an attractive city skyline view.",
    "forest": "A property photograph dominated by an attractive forest or woodland view.",
    "none": "A property photograph without a notable scenic view.",
}

VIEW_QUALITY = [
    "No scenic view, or an unattractive obstructed view.",
    "A limited or ordinary view with little visual appeal.",
    "An attractive scenic view that would improve a property listing.",
    "A spectacular, expansive, unobstructed view that could be the property's main attraction.",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, default=Path("data/demo-images"))
    parser.add_argument("--model", default="google/siglip2-base-patch16-224")
    args = parser.parse_args()

    manifest = json.loads((args.directory / "manifest.json").read_text())
    scorer = VisionScorer(model_name=args.model, aggregation="mean")

    print("image\ttype\ttype_score\tview_quality\tquality_confidence")
    for item in manifest:
        image = args.directory / item["file"]
        kind = scorer.choice(
            image,
            VIEW_TYPES,
            instructions="Classify only what is visibly present; ignore filenames and metadata.",
        )
        quality = scorer.score(
            image,
            VIEW_QUALITY,
            instructions="Rate only the visible view, not the room, property, description, or location.",
        )
        print(
            f"{item['slug']}\t{kind.choice}\t{kind.probabilities[kind.choice]:.3f}"
            f"\t{quality.score:.3f}\t{quality.confidence:.3f}"
        )


if __name__ == "__main__":
    main()
