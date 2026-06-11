import json
from app.config.rabbitmq_config import create_connection

def send_label(queue_name: str, message: dict):
    """
    Send a JSON message to RabbitMQ using your centralized connection config.
    """
    connection = create_connection()
    channel = connection.channel()

    channel.queue_declare(queue=queue_name, durable=True)

    channel.basic_publish(
        exchange="",
        routing_key=queue_name,
        body=json.dumps(message).encode("utf-8"),
        properties=None
    )

    connection.close()
    print(f"Sent message to {queue_name}: {message}")