import json
import base64
import pytest
from unittest.mock import MagicMock, patch

# conftest.py already injected mocks for person_model_loader and inference_person,
# so this import is safe — consumer.py's module-level instantiation uses those mocks.
import app.messiging.consumer as consumer_module
from app.messiging.consumer import callback, start_consumer


def _make_body(filename="test.jpg"):
    payload = {
        "fileName": filename,
        "data": base64.b64encode(b"fake_image_bytes").decode(),
    }
    return json.dumps(payload).encode()


class TestCallback:

    @patch("app.messiging.consumer.send_label")
    @patch("app.messiging.consumer.person_detector")
    def test_successful_prediction_sends_label_response(self, mock_detector, mock_send):
        mock_detector.predict_from_bytes.return_value = {
            "label": "Alice",
            "confidence": 0.95,
        }
        ch = MagicMock()
        method = MagicMock()
        method.delivery_tag = 1

        callback(ch, method, None, _make_body())

        mock_send.assert_called_once_with(
            "person_response_queue",
            {"fileName": "test.jpg", "labels": ["Alice"]},
        )

    @patch("app.messiging.consumer.send_label")
    @patch("app.messiging.consumer.person_detector")
    def test_successful_prediction_acks_message(self, mock_detector, mock_send):
        mock_detector.predict_from_bytes.return_value = {
            "label": "Bob",
            "confidence": 0.80,
        }
        ch = MagicMock()
        method = MagicMock()
        method.delivery_tag = 42

        callback(ch, method, None, _make_body())

        ch.basic_ack.assert_called_once_with(delivery_tag=42)

    @patch("app.messiging.consumer.send_label")
    @patch("app.messiging.consumer.person_detector")
    def test_prediction_error_sends_unknown_label(self, mock_detector, mock_send):
        mock_detector.predict_from_bytes.return_value = {"error": "No face detected"}
        ch = MagicMock()
        method = MagicMock()

        callback(ch, method, None, _make_body("photo.jpg"))

        mock_send.assert_called_once_with(
            "person_response_queue",
            {"fileName": "photo.jpg", "labels": ["Unknown"]},
        )

    @patch("app.messiging.consumer.send_label")
    @patch("app.messiging.consumer.person_detector")
    def test_exception_nacks_without_requeue(self, mock_detector, mock_send):
        mock_detector.predict_from_bytes.side_effect = RuntimeError("GPU failure")
        ch = MagicMock()
        method = MagicMock()
        method.delivery_tag = 7

        callback(ch, method, None, _make_body())

        ch.basic_nack.assert_called_once_with(delivery_tag=7, requeue=False)

    @patch("app.messiging.consumer.send_label")
    @patch("app.messiging.consumer.person_detector")
    def test_exception_does_not_ack(self, mock_detector, mock_send):
        mock_detector.predict_from_bytes.side_effect = ValueError("bad data")
        ch = MagicMock()
        method = MagicMock()

        callback(ch, method, None, _make_body())

        ch.basic_ack.assert_not_called()


class TestStartConsumer:

    def _run_start_consumer(self, mock_channel):
        """Run start_consumer and stop it after setup via KeyboardInterrupt."""
        mock_channel.start_consuming.side_effect = KeyboardInterrupt
        with pytest.raises(KeyboardInterrupt):
            start_consumer()

    @patch("app.messiging.consumer.create_connection")
    def test_declares_fanout_exchange(self, mock_conn):
        mock_channel = MagicMock()
        mock_conn.return_value.channel.return_value = mock_channel
        self._run_start_consumer(mock_channel)

        mock_channel.exchange_declare.assert_called_once_with(
            exchange="photo_exchange",
            exchange_type="fanout",
            durable=True,
        )

    @patch("app.messiging.consumer.create_connection")
    def test_declares_and_binds_service_queue(self, mock_conn):
        mock_channel = MagicMock()
        mock_conn.return_value.channel.return_value = mock_channel
        self._run_start_consumer(mock_channel)

        mock_channel.queue_declare.assert_called_once_with(
            queue="person_service_queue", durable=True
        )
        mock_channel.queue_bind.assert_called_once_with(
            exchange="photo_exchange", queue="person_service_queue"
        )

    @patch("app.messiging.consumer.create_connection")
    def test_sets_prefetch_count_to_one(self, mock_conn):
        mock_channel = MagicMock()
        mock_conn.return_value.channel.return_value = mock_channel
        self._run_start_consumer(mock_channel)

        mock_channel.basic_qos.assert_called_once_with(prefetch_count=1)

    @patch("app.messiging.consumer.create_connection")
    def test_registers_callback_and_starts_consuming(self, mock_conn):
        mock_channel = MagicMock()
        mock_conn.return_value.channel.return_value = mock_channel
        self._run_start_consumer(mock_channel)

        mock_channel.basic_consume.assert_called_once_with(
            queue="person_service_queue",
            on_message_callback=callback,
        )
        mock_channel.start_consuming.assert_called_once()
