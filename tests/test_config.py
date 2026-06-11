import pytest
from unittest.mock import patch, MagicMock


class TestCreateConnection:

    @patch("app.config.rabbitmq_config.pika")
    def test_returns_blocking_connection(self, mock_pika):
        mock_conn = MagicMock()
        mock_pika.BlockingConnection.return_value = mock_conn

        from app.config.rabbitmq_config import create_connection
        result = create_connection()

        assert result is mock_conn

    @patch("app.config.rabbitmq_config.pika")
    def test_uses_guest_credentials(self, mock_pika):
        from app.config.rabbitmq_config import create_connection
        create_connection()

        mock_pika.PlainCredentials.assert_called_once_with("guest", "guest")

    @patch("app.config.rabbitmq_config.pika")
    def test_connection_parameters_use_localhost_5672(self, mock_pika):
        from app.config.rabbitmq_config import create_connection
        create_connection()

        mock_pika.ConnectionParameters.assert_called_once_with(
            host="localhost",
            port=5672,
            credentials=mock_pika.PlainCredentials.return_value,
        )

    @patch("app.config.rabbitmq_config.pika")
    def test_blocking_connection_receives_parameters(self, mock_pika):
        from app.config.rabbitmq_config import create_connection
        create_connection()

        mock_pika.BlockingConnection.assert_called_once_with(
            mock_pika.ConnectionParameters.return_value
        )
