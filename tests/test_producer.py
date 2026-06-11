import json
import pytest
from unittest.mock import patch, MagicMock


class TestSendLabel:

    @patch("app.messiging.producer.create_connection")
    def test_declares_durable_queue(self, mock_create_conn):
        mock_channel = MagicMock()
        mock_create_conn.return_value.channel.return_value = mock_channel

        from app.messiging.producer import send_label
        send_label("test_queue", {"key": "value"})

        mock_channel.queue_declare.assert_called_once_with(
            queue="test_queue", durable=True
        )

    @patch("app.messiging.producer.create_connection")
    def test_publishes_json_encoded_message(self, mock_create_conn):
        mock_channel = MagicMock()
        mock_create_conn.return_value.channel.return_value = mock_channel

        from app.messiging.producer import send_label
        message = {"fileName": "test.jpg", "labels": ["Alice"]}
        send_label("response_queue", message)

        mock_channel.basic_publish.assert_called_once_with(
            exchange="",
            routing_key="response_queue",
            body=json.dumps(message).encode("utf-8"),
            properties=None,
        )

    @patch("app.messiging.producer.create_connection")
    def test_closes_connection_after_publish(self, mock_create_conn):
        mock_conn = MagicMock()
        mock_create_conn.return_value = mock_conn

        from app.messiging.producer import send_label
        send_label("queue", {"data": "test"})

        mock_conn.close.assert_called_once()

    @patch("app.messiging.producer.create_connection")
    def test_publishes_to_the_named_queue(self, mock_create_conn):
        mock_channel = MagicMock()
        mock_create_conn.return_value.channel.return_value = mock_channel

        from app.messiging.producer import send_label
        send_label("special_queue", {})

        _, kwargs = mock_channel.basic_publish.call_args
        assert kwargs["routing_key"] == "special_queue"
