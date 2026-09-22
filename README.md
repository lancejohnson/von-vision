# Von Vision

**Zero-shot, Von-style typed visual decisions with SigLIP/SigLIP2.**

Von Vision composes a Hugging Face SigLIP-family dual encoder with typed decision
primitives. It deliberately adds **no classifier or scoring head** and performs no
training: local image and runtime text embeddings are L2-normalized, compared using
the model's native logit scale/bias when present, then softmaxed over the supplied
candidates.

## Important limitation

The returned distributions are **relative softmax scores across the candidates in
that request**, not calibrated probabilities. Adding, removing, or rewording a
candidate can change every score. They are useful for zero-shot ranking and routing;
do not use them as a confidence guarantee or safety threshold without validation on
your own data.

## Install

```bash
pip install 'von-vision[vision]'
# or, from this checkout:
pip install -e '.[vision]'
```

The default model is `google/siglip2-base-patch16-224`. Use any compatible local or
Hugging Face SigLIP/SigLIP2 model with `VisionScorer(model_name=...)`. Model downloads
happen lazily on the first request; tests can inject embedding providers and make no
network request. The retained upstream text package is available as `von-vision[von]`;
install `von-vision[vision,von]` to use both stacks.

## Python

```python
from von_vision import VisionScorer

vision = VisionScorer(aggregation="mean")
view = vision.choice(
    ["listing-front.jpg", "listing-balcony.jpg"],
    {
        "water": "A photograph with an expansive unobstructed water view.",
        "mountains": "A photograph with an expansive unobstructed mountain view.",
        "none": "A photograph without a notable scenic view.",
    },
    instructions="Which visual description best fits these listing photos?",
)
print(view.choice, view.probabilities, view.confidence)

beauty = vision.score(
    ["listing-front.jpg", "listing-balcony.jpg"],
    ["poor or obstructed view", "pleasant view", "exceptional unobstructed scenic view"],
    instructions="Rate the quality of the visible view.",
)
print(beauty.score)  # normalized 0.0–1.0; relative to these three levels

has_view = vision.noul(
    "listing-balcony.jpg",
    "Does this image show an unobstructed scenic view?",
)
print(has_view.noul, has_view.probabilities)
```

Image inputs are local paths or `PIL.Image.Image` objects. A `VisionScorer` caches
normalized embeddings for local paths by path/mtime, so subsequent criteria reuse
image features. PIL images are deliberately not cached because they are mutable;
save them to a local path when reusable caching is required. `aggregation="mean"`
averages per-image candidate logits; `aggregation="max"` uses the strongest image
for each candidate.

## CLI

```bash
von-vision balcony.jpg terrace.jpg \
  --instructions "Which view is visible?" \
  --option water="Unobstructed water view" \
  --option city="Expansive city skyline view" \
  --option none="No notable scenic view" \
  --aggregation mean
```

## Real-image demo

Download six openly licensed Wikimedia Commons images (water, mountains, skyline,
forest, bedroom, and parking lot), then score them with the default SigLIP2 model:

```bash
uv run --extra vision python examples/download_demo_images.py
uv run --extra vision python examples/score_demo_images.py
```

The downloader writes the images and a source/license manifest under the ignored
`data/demo-images/` directory. The scoring script tests dynamic view classification
and an ordered scenic-view-quality score without using filenames or descriptions as
model input.

## Scope

This project does not fetch image URLs, decode video, train a model, perform visual
click grounding, or infer a universal aesthetic truth. For video, extract frames with
your existing decoder, then score selected frames; for web control, use a separate
region/coordinate-grounding component.

## Attribution and license

Von Vision is an Apache-2.0 fork/composition derived from
[wfzyx/von](https://github.com/wfzyx/von); its retained `von` package and Apache-2.0
license remain in this repository. The visual encoder integration targets Google's
[SigLIP and SigLIP2](https://huggingface.co/docs/transformers/model_doc/siglip)
models via Hugging Face Transformers. SigLIP model weights have their own model-card
terms and must be reviewed separately from this repository's Apache-2.0 code license.
