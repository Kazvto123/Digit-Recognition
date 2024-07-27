from flask import Flask, render_template, request, jsonify
import numpy as np
import tensorflow as tf
import cv2
import base64

app = Flask(__name__)
model = tf.keras.models.load_model('handwritten_model.h5')

def preprocess_image(img_data):
    # Decode base64 image data
    nparr = np.frombuffer(base64.b64decode(img_data), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    # Resize to 28x28 pixels
    img = cv2.resize(img, (28, 28))
    # Invert colors
    img = np.invert(img)
    # Normalize the image
    img = img / 255.0
    # Reshape to match the model input
    img = img.reshape(1, 28, 28, 1)
    return img

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    img_str = data['image']
    img = preprocess_image(img_str)
    prediction = model.predict(img)
    digit = int(np.argmax(prediction))
    return jsonify({'digit': digit})

if __name__ == '__main__':
    app.run(debug=True)
