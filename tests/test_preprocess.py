from pathlib import Path

from inference_lab.preprocess import load_image


def test_load_image_returns_nchw_tensor() -> None:
    sample = Path("sample_images/gradient.ppm")
    output = load_image(sample, image_size=(224, 224))

    assert output.shape == (1, 3, 224, 224)
    assert str(output.dtype) == "float32"
