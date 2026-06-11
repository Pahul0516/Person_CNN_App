import json
import base64
from app.config.rabbitmq_config import create_connection
from app.messiging.producer import send_label
from app.model.person.inference_person import PersonRecognizer
from app.model.person.person_model_loader import PersonCNNModel

EXCHANGE = "photo_exchange"
SERVICE_QUEUE = "person_service_queue"

# 1. SETUP THE MODEL (Done once at startup)
MODEL_PATH = '../notebooks/ResNet50_lfw50/Checkpoints/PersonCNN_ResNet50_lfw50_FT_epoch_206_val_acc_0.8500.keras'
JSON_PATH = '../notebooks/classes.json'

# Initialize the loader and recognizer
# We set a default zoom here, but we can override it later if needed
loader = PersonCNNModel(MODEL_PATH, JSON_PATH)
person_detector = PersonRecognizer(loader, tflite_path='../notebooks/face_detector.tflite', default_zoom=0.3)

def callback(ch, method, properties, body):
    try:
        message = json.loads(body.decode())
        file_name = message["fileName"]
        image_bytes = base64.b64decode(message["data"])

        # 2. RUN PREDICTION
        # Since your class returns a dict with 'label' and 'confidence'
        result = person_detector.predict_from_bytes(image_bytes)

        if "error" in result:
            print(f"Prediction failed for {file_name}: {result['error']}")
            detected_labels = ["Unknown"]
        else:
            # We wrap the single label in a list to keep your existing logic compatible
            detected_labels = [result["label"]]
            print(f"{file_name} → {result['label']} ({result['confidence']*100:.2f}%)")

        # 3. SEND RESPONSE
        response_queue = "person_response_queue"
        send_label(response_queue, {"fileName": file_name, "labels": detected_labels})

        ch.basic_ack(delivery_tag=method.delivery_tag)

    except Exception as e:
        print("Error processing message:", e)
        # Requeue=False prevents an infinite loop if the image is corrupted
        ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)

def start_consumer():
    connection = create_connection()
    channel = connection.channel()

    channel.exchange_declare(
        exchange=EXCHANGE,
        exchange_type="fanout",
        durable=True
    )

    channel.queue_declare(queue=SERVICE_QUEUE, durable=True)
    channel.queue_bind(exchange=EXCHANGE, queue=SERVICE_QUEUE)

    # Performance Tip: Process only 1 message at a time to avoid RAM spikes
    channel.basic_qos(prefetch_count=1)

    channel.basic_consume(
        queue=SERVICE_QUEUE,
        on_message_callback=callback
    )

    print("Waiting for messages...")
    channel.start_consuming()