import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

class PersonRecognizer:
    def __init__(self, model_loader, tflite_path: str = 'notebooks/face_detector.tflite', default_zoom: float = 0.3):
        self.model_wrapper = model_loader
        self.model = self.model_wrapper.model
        self.categories = self.model_wrapper.categories

        # Adjustable Variable
        self.zoom_out = default_zoom

        # Initialize MediaPipe Task
        base_options = python.BaseOptions(model_asset_path=tflite_path)
        options = vision.FaceDetectorOptions(base_options=base_options)
        self.detector = vision.FaceDetector.create_from_options(options)

    def _apply_zoom_logic(self, bbox, img_shape, zoom_factor):
        h_img, w_img, _ = img_shape
        dw = int(bbox.width * zoom_factor)
        dh = int(bbox.height * zoom_factor)

        nx = max(0, bbox.origin_x - dw)
        ny = max(0, bbox.origin_y - dh)
        nw = min(w_img - nx, bbox.width + (2 * dw))
        nh = min(h_img - ny, bbox.height + (2 * dh))

        return nx, ny, nw, nh

    def predict_from_bytes(self, image_bytes: bytes, zoom_override: float = None):
        """
        New method to handle raw bytes (from API or RabbitMQ)
        """
        # Convert bytes to numpy array for OpenCV
        nparr = np.frombuffer(image_bytes, np.uint8)
        image_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image_bgr is None:
            return {"error": "Could not decode image bytes"}

        return self._process_and_predict(image_bgr, zoom_override)

    def predict_from_path(self, img_path: str, zoom_override: float = None):
        image_bgr = cv2.imread(img_path)
        if image_bgr is None:
            return {"error": "Image not found at path"}

        return self._process_and_predict(image_bgr, zoom_override)

    def _process_and_predict(self, image_bgr, zoom_override):
        """
        Internal shared logic for both Path and Bytes inputs
        """
        current_zoom = zoom_override if zoom_override is not None else self.zoom_out
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        # MediaPipe Detection
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        detection_result = self.detector.detect(mp_image)

        if not detection_result.detections:
            return {"error": "No face detected"}

        # Apply Zoom & Crop
        bbox = detection_result.detections[0].bounding_box
        nx, ny, nw, nh = self._apply_zoom_logic(bbox, image_rgb.shape, current_zoom)
        face_crop = image_rgb[ny:ny + nh, nx:nx + nw]

        # Preprocess & Predict
        face_resized = cv2.resize(face_crop, (224, 224))
        img_array = tf.keras.preprocessing.image.img_to_array(face_resized)
        img_array = np.expand_dims(img_array, axis=0)
        processed_img = tf.keras.applications.resnet50.preprocess_input(img_array)

        predictions = self.model.predict(processed_img, verbose=0)[0]
        predicted_index = np.argmax(predictions)

        return {
            "label": self.categories[predicted_index],
            "confidence": float(predictions[predicted_index]),
            "zoom_used": current_zoom,
            "box": [nx, ny, nw, nh]  # Coordinates of the zoomed face
        }