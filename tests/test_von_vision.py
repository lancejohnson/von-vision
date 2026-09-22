from math import isclose
import sys
from types import SimpleNamespace

import pytest
from PIL import Image

from von_vision import VisionScorer


class Providers:
    def __init__(self):
        self.image_calls = 0
        self.text_calls = 0
        self.last_prompts = []

    def images(self, images):
        self.image_calls += 1
        return [[1.0, 0.0] if image.getpixel((0, 0))[0] else [0.0, 1.0] for image in images]

    def text(self, prompts):
        self.text_calls += 1
        self.last_prompts = list(prompts)
        output = []
        for prompt in prompts:
            if "water" in prompt or "yes" in prompt or "excellent" in prompt:
                output.append([1.0, 0.0])
            elif "no" in prompt or "poor" in prompt:
                output.append([0.0, 1.0])
            else:
                output.append([0.5, 0.5])
        return output


def scorer(**kwargs):
    providers = Providers()
    return VisionScorer(image_embedding_provider=providers.images, text_embedding_provider=providers.text, **kwargs), providers


def red():
    return Image.new("RGB", (2, 2), (255, 0, 0))


def blue():
    return Image.new("RGB", (2, 2), (0, 0, 255))


def test_choice_returns_competing_relative_distribution():
    vision, _ = scorer()
    result = vision.choice(red(), {"water": "water view", "city": "city view"}, instructions="visible")
    assert result.choice == "water"
    assert set(result.probabilities) == {"water", "city"}
    assert isclose(sum(result.probabilities.values()), 1.0)
    assert result.probability_kind == "relative_softmax"
    assert result.confidence > 0


def test_noul_uses_explicit_yes_no_candidates():
    vision, providers = scorer()
    result = vision.noul(red(), "Does this show a scenic view?")
    assert result.noul > 0.5
    assert result.probabilities["true"] == result.noul
    assert any("yes" in prompt for prompt in providers.last_prompts)
    assert any("no" in prompt for prompt in providers.last_prompts)
    assert providers.text_calls == 1


def test_score_is_probability_weighted_and_normalized():
    vision, _ = scorer()
    result = vision.score(red(), ["poor view", "average view", "excellent water view"], instructions="Rate the view")
    assert 0.0 <= result.score <= 1.0
    assert result.score > 0.5
    assert result.legend["2"] == "excellent water view"
    assert isclose(sum(result.probabilities.values()), 1.0)


def test_mean_and_max_multi_image_aggregation_differ():
    def images(items):
        return [[1.0, 0.0] if image.getpixel((0, 0))[0] else [1.0, 1.0] for image in items]
    def texts(prompts):
        return [[1.0, 0.0] if "water" in prompt else [0.0, 1.0] for prompt in prompts]
    vision = VisionScorer(image_embedding_provider=images, text_embedding_provider=texts)
    criteria = {"water": "water", "none": "no view"}
    mean = vision.choice([red(), blue()], criteria, aggregation="mean")
    maximum = vision.choice([red(), blue()], criteria, aggregation="max")
    assert not isclose(mean.probabilities["water"], maximum.probabilities["water"])


def test_local_path_is_accepted(tmp_path):
    path = tmp_path / "view.png"
    red().save(path)
    vision, _ = scorer()
    result = vision.choice(path, {"water": "water", "city": "city"})
    assert result.choice == "water"


def test_local_path_embeddings_are_cached_between_questions(tmp_path):
    path = tmp_path / "view.png"
    red().save(path)
    vision, providers = scorer()
    vision.choice(path, {"water": "water", "city": "city"})
    vision.score(path, ["poor", "excellent water"])
    assert providers.image_calls == 1
    assert vision.cached_image_count == 1
    vision.clear_image_cache()
    assert vision.cached_image_count == 0


def test_mutated_pil_image_is_not_served_from_cache():
    vision, providers = scorer()
    image = red()
    first = vision.choice(image, {"water": "water", "city": "city"})
    image.paste((0, 0, 255), (0, 0, 2, 2))
    second = vision.choice(image, {"water": "water", "city": "city"})
    assert first.choice == "water"
    assert second.choice == "city"
    assert providers.image_calls == 2
    assert vision.cached_image_count == 0


def test_native_scale_and_bias_are_applied_when_available():
    class Scalar:
        def __init__(self, value): self.value = value
        def item(self): return self.value
        def detach(self): return self
        def cpu(self): return self

    class FakeModel:
        logit_scale = Scalar(0.6931471805599453)  # ln(2)
        logit_bias = Scalar(1.0)

    vision, _ = scorer(model=FakeModel(), processor=object())
    assert isclose(vision._scaled_similarity([1, 0], [1, 0]), 3.0)


def test_native_text_embeddings_use_fixed_length_padding(monkeypatch):
    calls = []

    class Processor:
        def __call__(self, **kwargs):
            calls.append(kwargs)
            return {}

    class Tensor:
        def detach(self): return self
        def cpu(self): return self
        def tolist(self): return [[1.0, 0.0]]

    class Model:
        def get_text_features(self, **_): return Tensor()

    class NoGrad:
        def __enter__(self): return None
        def __exit__(self, *_): return False

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(no_grad=NoGrad))
    vision = VisionScorer(model=Model(), processor=Processor(), device="cpu")
    vision._text_embeddings(["a scenic view"])
    assert calls == [{"text": ["a scenic view"], "padding": "max_length", "return_tensors": "pt"}]


def test_model_and_processor_must_be_supplied_together():
    with pytest.raises(ValueError, match="model and processor together"):
        VisionScorer(model=object(), image_embedding_provider=lambda _: [], text_embedding_provider=lambda _: [])
    with pytest.raises(ValueError, match="model and processor together"):
        VisionScorer(processor=object(), image_embedding_provider=lambda _: [], text_embedding_provider=lambda _: [])


@pytest.mark.parametrize("call", [
    lambda vision: vision.choice([], {"a": "a"}),
    lambda vision: vision.choice(red(), {}),
    lambda vision: vision.score(red(), ["only one"]),
    lambda vision: vision.noul(red(), ""),
    lambda vision: vision.choice(red(), {"a": "a"}, aggregation="median"),
])
def test_invalid_inputs_raise_clear_errors(call):
    vision, _ = scorer()
    with pytest.raises(ValueError):
        call(vision)


def test_missing_path_and_non_image_input_are_rejected():
    vision, _ = scorer()
    with pytest.raises(ValueError, match="does not exist"):
        vision.choice("/definitely/not/a/photo.jpg", {"a": "a"})
    with pytest.raises(TypeError):
        vision.choice([object()], {"a": "a"})
