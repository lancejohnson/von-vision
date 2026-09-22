"""Zero-shot, typed visual decisions built from a SigLIP-style dual encoder.

This module deliberately has no learned decision head.  Candidate prompts and images
are embedded by the same dual encoder, then compared in its shared embedding space.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import exp
from pathlib import Path
from typing import Any, Callable, Iterable, Literal, Mapping, Sequence

ImageInput = str | Path | Any
Aggregation = Literal["mean", "max"]
EmbeddingProvider = Callable[[Sequence[Any]], Sequence[Sequence[float]]]


@dataclass(frozen=True)
class ChoiceResult:
    choice: str
    probabilities: dict[str, float]
    confidence: float
    probability_kind: str = "relative_softmax"


@dataclass(frozen=True)
class NoulResult:
    noul: float
    confidence: float
    probabilities: dict[str, float]
    probability_kind: str = "relative_softmax"


@dataclass(frozen=True)
class ScoreResult:
    score: float
    confidence: float
    legend: dict[str, str]
    probabilities: dict[str, float]
    probability_kind: str = "relative_softmax"


def _normalize(vector: Sequence[float]) -> list[float]:
    values = [float(v) for v in vector]
    norm = sum(v * v for v in values) ** 0.5
    if norm == 0:
        raise ValueError("Encoder returned a zero-length embedding.")
    return [v / norm for v in values]


def _softmax(logits: Sequence[float]) -> list[float]:
    if not logits:
        raise ValueError("At least one candidate is required.")
    maximum = max(logits)
    weights = [exp(value - maximum) for value in logits]
    total = sum(weights)
    return [weight / total for weight in weights]


class VisionScorer:
    """Score local images against runtime natural-language criteria.

    ``probabilities`` are a softmax *relative to the candidates passed to that
    request*. They are not calibrated posterior probabilities.  Supply embedding
    providers in tests or use the lazy default SigLIP/SigLIP2 loader in production.
    """

    def __init__(
        self,
        model_name: str = "google/siglip2-base-patch16-224",
        *,
        aggregation: Aggregation = "mean",
        device: str | None = None,
        model: Any | None = None,
        processor: Any | None = None,
        image_embedding_provider: EmbeddingProvider | None = None,
        text_embedding_provider: EmbeddingProvider | None = None,
    ) -> None:
        self._validate_aggregation(aggregation)
        if (image_embedding_provider is None) != (text_embedding_provider is None):
            raise ValueError("Provide both image_embedding_provider and text_embedding_provider.")
        if (model is None) != (processor is None):
            raise ValueError("Provide model and processor together, or neither.")
        self.model_name = model_name
        self.aggregation = aggregation
        self.device = device
        self._model = model
        self._processor = processor
        self._image_provider = image_embedding_provider
        self._text_provider = text_embedding_provider
        self._image_cache: dict[tuple[Any, ...], list[float]] = {}

    @property
    def cached_image_count(self) -> int:
        """Number of image embeddings retained by this scorer instance."""
        return len(self._image_cache)

    def clear_image_cache(self) -> None:
        self._image_cache.clear()

    def choice(
        self,
        images: ImageInput | Sequence[ImageInput],
        criteria: Mapping[str, str | None],
        *,
        instructions: str = "",
        aggregation: Aggregation | None = None,
    ) -> ChoiceResult:
        if not criteria:
            raise ValueError("choice requires at least one criterion.")
        keys = list(criteria)
        if any(not isinstance(key, str) or not key.strip() for key in keys):
            raise ValueError("choice criterion keys must be non-empty strings.")
        prompts = [self._prompt(instructions, criteria[key] or key) for key in keys]
        probabilities = self._probabilities(images, prompts, aggregation)
        distribution = dict(zip(keys, probabilities))
        best = max(range(len(keys)), key=probabilities.__getitem__)
        return ChoiceResult(keys[best], distribution, self._confidence(probabilities))

    def noul(
        self,
        images: ImageInput | Sequence[ImageInput],
        instructions: str,
        *,
        positive: str | None = None,
        negative: str | None = None,
        aggregation: Aggregation | None = None,
    ) -> NoulResult:
        if not isinstance(instructions, str) or not instructions.strip():
            raise ValueError("noul requires non-empty instructions.")
        positive = positive or f"{instructions.strip()} Answer: yes."
        negative = negative or f"{instructions.strip()} Answer: no."
        if not positive.strip() or not negative.strip():
            raise ValueError("noul alternatives must be non-empty strings.")
        values = self._probabilities(images, [positive, negative], aggregation)
        return NoulResult(values[0], self._confidence(values), {"true": values[0], "false": values[1]})

    def score(
        self,
        images: ImageInput | Sequence[ImageInput],
        criteria: Sequence[str],
        *,
        instructions: str = "",
        aggregation: Aggregation | None = None,
    ) -> ScoreResult:
        if len(criteria) < 2:
            raise ValueError("score requires at least two ordered criteria.")
        if any(not isinstance(level, str) or not level.strip() for level in criteria):
            raise ValueError("score criteria must be non-empty strings.")
        prompts = [self._prompt(instructions, level) for level in criteria]
        values = self._probabilities(images, prompts, aggregation)
        denominator = len(values) - 1
        normalized = sum(index * value for index, value in enumerate(values)) / denominator
        legend = {str(index): level for index, level in enumerate(criteria)}
        distribution = {str(index): value for index, value in enumerate(values)}
        return ScoreResult(normalized, self._confidence(values), legend, distribution)

    def _probabilities(self, images: ImageInput | Sequence[ImageInput], prompts: Sequence[str], aggregation: Aggregation | None) -> list[float]:
        mode = aggregation or self.aggregation
        self._validate_aggregation(mode)
        loaded = self._coerce_images(images)
        image_embeddings = [self._image_embedding(image) for image in loaded]
        text_embeddings = [_normalize(vector) for vector in self._text_embeddings(prompts)]
        if any(len(image_embeddings[0]) != len(text) for text in text_embeddings):
            raise ValueError("Image and text embeddings must have the same dimensions.")
        per_image = [[self._scaled_similarity(image, text) for text in text_embeddings] for image in image_embeddings]
        if mode == "mean":
            logits = [sum(row[index] for row in per_image) / len(per_image) for index in range(len(prompts))]
        else:
            logits = [max(row[index] for row in per_image) for index in range(len(prompts))]
        return _softmax(logits)

    @staticmethod
    def _validate_aggregation(mode: str) -> None:
        if mode not in ("mean", "max"):
            raise ValueError("aggregation must be 'mean' or 'max'.")

    @staticmethod
    def _confidence(values: Sequence[float]) -> float:
        ordered = sorted(values, reverse=True)
        return ordered[0] - (ordered[1] if len(ordered) > 1 else 0.0)

    @staticmethod
    def _prompt(instructions: str, criterion: str) -> str:
        return f"{instructions.strip()} {criterion.strip()}".strip()

    def _coerce_images(self, images: ImageInput | Sequence[ImageInput]) -> list[Any]:
        if isinstance(images, (str, Path)) or hasattr(images, "convert"):
            candidates = [images]
        elif isinstance(images, Sequence):
            candidates = list(images)
        else:
            raise TypeError("images must be a local path, PIL image, or a sequence of either.")
        if not candidates:
            raise ValueError("At least one image is required.")
        return [self._load_image(image) for image in candidates]

    def _load_image(self, image: ImageInput) -> Any:
        if isinstance(image, (str, Path)):
            path = Path(image).expanduser().resolve()
            if not path.is_file():
                raise ValueError(f"Image path does not exist or is not a file: {path}")
            try:
                from PIL import Image
            except ImportError as error:
                raise ImportError("Install von-vision[vision] to use local image paths.") from error
            with Image.open(path) as opened:
                return opened.convert("RGB").copy(), ("path", str(path), path.stat().st_mtime_ns)
        if hasattr(image, "convert"):
            # PIL images are mutable and object ids may be reused after collection.
            # Do not cache them; callers needing reuse can pass an immutable file path.
            return image.convert("RGB"), None
        raise TypeError("Each image must be a local path or PIL.Image.Image.")

    def _image_embedding(self, loaded: tuple[Any, tuple[Any, ...] | None]) -> list[float]:
        image, key = loaded
        if key is None:
            values = self._image_embeddings([image])
            if len(values) != 1:
                raise ValueError("Image encoder returned an unexpected number of embeddings.")
            return _normalize(values[0])
        if key not in self._image_cache:
            values = self._image_embeddings([image])
            if len(values) != 1:
                raise ValueError("Image encoder returned an unexpected number of embeddings.")
            self._image_cache[key] = _normalize(values[0])
        return self._image_cache[key]

    def _image_embeddings(self, images: Sequence[Any]) -> Sequence[Sequence[float]]:
        if self._image_provider is not None:
            return self._image_provider(images)
        self._ensure_model()
        import torch
        inputs = self._processor(images=list(images), return_tensors="pt")
        inputs = {key: value.to(self._resolved_device()) for key, value in inputs.items()}
        with torch.no_grad():
            return self._model.get_image_features(**inputs).detach().cpu().tolist()

    def _text_embeddings(self, prompts: Sequence[str]) -> Sequence[Sequence[float]]:
        if self._text_provider is not None:
            return self._text_provider(prompts)
        self._ensure_model()
        import torch
        inputs = self._processor(text=list(prompts), padding="max_length", return_tensors="pt")
        inputs = {key: value.to(self._resolved_device()) for key, value in inputs.items()}
        with torch.no_grad():
            return self._model.get_text_features(**inputs).detach().cpu().tolist()

    def _ensure_model(self) -> None:
        if self._model is not None and self._processor is not None:
            return
        try:
            import torch
            from transformers import AutoModel, AutoProcessor
        except ImportError as error:
            raise ImportError("Install von-vision[vision] to run SigLIP/SigLIP2 inference.") from error
        self._processor = AutoProcessor.from_pretrained(self.model_name)
        self._model = AutoModel.from_pretrained(self.model_name).to(self._resolved_device()).eval()

    def _resolved_device(self) -> str:
        if self.device:
            return self.device
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"

    def _scaled_similarity(self, image: Sequence[float], text: Sequence[float]) -> float:
        value = sum(a * b for a, b in zip(image, text))
        model = self._model
        if model is None:
            return value
        scale = getattr(model, "logit_scale", None)
        bias = getattr(model, "logit_bias", None)
        if scale is not None:
            scale = float(scale.detach().cpu().item() if hasattr(scale, "detach") else scale)
            value *= exp(scale)
        if bias is not None:
            bias = float(bias.detach().cpu().item() if hasattr(bias, "detach") else bias)
            value += bias
        return value
