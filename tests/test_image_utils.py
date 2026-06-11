import pytest
from io import BytesIO
from PIL import Image

from app.utils.image_utils import load_image_from_bytes


def _make_image_bytes(mode="RGB", size=(60, 40), fmt="JPEG"):
    img = Image.new(mode, size, color=(100, 150, 200) if mode in ("RGB", "RGBA") else 128)
    buf = BytesIO()
    if fmt == "JPEG" and mode != "RGB":
        img = img.convert("RGB")
    img.save(buf, format=fmt)
    return buf.getvalue()


class TestLoadImageFromBytes:

    def test_returns_pil_image(self):
        result = load_image_from_bytes(_make_image_bytes())
        assert isinstance(result, Image.Image)

    def test_result_is_rgb_mode(self):
        result = load_image_from_bytes(_make_image_bytes(mode="RGB"))
        assert result.mode == "RGB"

    def test_rgba_png_converted_to_rgb(self):
        result = load_image_from_bytes(_make_image_bytes(mode="RGBA", fmt="PNG"))
        assert result.mode == "RGB"

    def test_grayscale_png_converted_to_rgb(self):
        result = load_image_from_bytes(_make_image_bytes(mode="L", fmt="PNG"))
        assert result.mode == "RGB"

    def test_preserves_image_dimensions(self):
        result = load_image_from_bytes(_make_image_bytes(size=(120, 80)))
        assert result.size == (120, 80)

    def test_loads_png_format(self):
        result = load_image_from_bytes(_make_image_bytes(fmt="PNG"))
        assert isinstance(result, Image.Image)
        assert result.mode == "RGB"
