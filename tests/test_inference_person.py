import sys
import numpy as np
import pytest
from unittest.mock import MagicMock, patch

# Remove the conftest stub so this file loads the real module.
sys.modules.pop("app.model.person.inference_person", None)

from app.model.person.inference_person import PersonRecognizer


class FakeBbox:
    def __init__(self, x, y, w, h):
        self.origin_x = x
        self.origin_y = y
        self.width = w
        self.height = h


@pytest.fixture
def recognizer():
    """PersonRecognizer with MediaPipe/tflite dependencies mocked out."""
    with patch("app.model.person.inference_person.vision"), \
         patch("app.model.person.inference_person.python"):
        mock_loader = MagicMock()
        mock_loader.model = MagicMock()
        mock_loader.categories = ["Alice", "Bob"]
        yield PersonRecognizer(mock_loader, tflite_path="/fake/detector.tflite")


class TestApplyZoomLogic:

    def test_basic_zoom_expands_bbox(self, recognizer):
        bbox = FakeBbox(x=50, y=50, w=100, h=100)
        nx, ny, nw, nh = recognizer._apply_zoom_logic(bbox, (200, 200, 3), 0.1)

        assert nx == 40
        assert ny == 40
        assert nw == 120
        assert nh == 120

    def test_origin_clamped_to_zero_at_image_edge(self, recognizer):
        bbox = FakeBbox(x=5, y=5, w=40, h=40)
        nx, ny, nw, nh = recognizer._apply_zoom_logic(bbox, (100, 100, 3), 0.5)

        assert nx == 0
        assert ny == 0

    def test_width_clamped_to_image_boundary(self, recognizer):
        # bbox near right edge — expanded width would exceed image width
        bbox = FakeBbox(x=80, y=80, w=30, h=30)
        nx, ny, nw, nh = recognizer._apply_zoom_logic(bbox, (100, 100, 3), 0.5)

        # nw must not exceed (w_img - nx)
        assert nx + nw <= 100
        assert ny + nh <= 100


class TestPredictFromBytes:

    @patch("app.model.person.inference_person.cv2")
    def test_returns_error_when_image_cannot_be_decoded(self, mock_cv2, recognizer):
        mock_cv2.imdecode.return_value = None

        result = recognizer.predict_from_bytes(b"not_an_image")

        assert result == {"error": "Could not decode image bytes"}

    @patch("app.model.person.inference_person.cv2")
    def test_delegates_to_process_and_predict(self, mock_cv2, recognizer):
        fake_img = np.zeros((100, 100, 3), dtype=np.uint8)
        mock_cv2.imdecode.return_value = fake_img
        mock_cv2.IMREAD_COLOR = 1

        with patch.object(
            recognizer, "_process_and_predict", return_value={"label": "Alice", "confidence": 0.9}
        ) as mock_process:
            result = recognizer.predict_from_bytes(b"fake_bytes")

        mock_process.assert_called_once_with(fake_img, None)
        assert result["label"] == "Alice"


class TestPredictFromPath:

    @patch("app.model.person.inference_person.cv2")
    def test_returns_error_when_image_not_found(self, mock_cv2, recognizer):
        mock_cv2.imread.return_value = None

        result = recognizer.predict_from_path("/nonexistent/photo.jpg")

        assert result == {"error": "Image not found at path"}

    @patch("app.model.person.inference_person.cv2")
    def test_delegates_to_process_and_predict(self, mock_cv2, recognizer):
        fake_img = np.zeros((100, 100, 3), dtype=np.uint8)
        mock_cv2.imread.return_value = fake_img

        with patch.object(
            recognizer, "_process_and_predict", return_value={"label": "Bob", "confidence": 0.8}
        ) as mock_process:
            result = recognizer.predict_from_path("/path/to/photo.jpg")

        mock_process.assert_called_once_with(fake_img, None)
        assert result["label"] == "Bob"


class TestProcessAndPredict:

    @patch("app.model.person.inference_person.mp")
    @patch("app.model.person.inference_person.cv2")
    def test_returns_error_when_no_face_detected(self, mock_cv2, mock_mp, recognizer):
        fake_img = np.zeros((100, 100, 3), dtype=np.uint8)
        mock_cv2.cvtColor.return_value = fake_img
        mock_cv2.COLOR_BGR2RGB = 4

        recognizer.detector.detect.return_value.detections = []

        result = recognizer._process_and_predict(fake_img, None)

        assert result == {"error": "No face detected"}
