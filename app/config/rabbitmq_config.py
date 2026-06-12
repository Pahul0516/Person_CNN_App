import pika


def create_connection():
    credentials = pika.PlainCredentials("guest", "guest")

    parameters = pika.ConnectionParameters(
        host="localhost",
        port=5672,
        credentials=credentials
    )

    return pika.BlockingConnection(parameters)
