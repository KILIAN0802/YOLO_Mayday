from flask import Flask, request, jsonify, send_file
import os
import uuid
import cv2
import base64
from inference_core import load_yolo_model, predict_image

app = Flask(__name__)

# Load the model once when the app starts
MODEL_PATH = os.environ.get("MODEL_PATH", "/app/weights/best.pt")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model file not found at {MODEL_PATH}. Put best.pt in weights/ before building the image or set MODEL_PATH."
    )

yolo_model = load_yolo_model(MODEL_PATH)

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file:
        filename = f"{uuid.uuid4()}_{file.filename}"
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        results = predict_image(yolo_model, filepath)

        # Clean up the uploaded image
        os.remove(filepath)

        # Encode the annotated image to base64 if available
        if 'AnnotatedImage' in results and results['AnnotatedImage'] is not None:
            annotated_img_array = results.pop('AnnotatedImage')
            _, buffer = cv2.imencode('.jpg', annotated_img_array)
            encoded_image = base64.b64encode(buffer).decode('utf-8')
            results['annotated_image_base64'] = encoded_image

        return jsonify(results), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)