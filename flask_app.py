from flask import Flask, render_template, request, jsonify, send_file
import cv2
import numpy as np
from inference_sdk import InferenceHTTPClient
from deepfake_detection import predict_deepfake

app = Flask(__name__)

# Initialize the InferenceHTTPClient
CLIENT = InferenceHTTPClient(
    api_url="https://classify.roboflow.com",
    api_key="8RSJzoEweFB7NxxNK6fg"
)

# Function to perform inference and return results
def perform_inference(image):
    # Perform inference on the input image
    result = CLIENT.infer(image, model_id="face-anti-spoofing-icbck/1")

    # Extract predictions from the result
    predictions = result['predictions']
    if predictions:
        # Sort predictions by confidence score in descending order
        predictions.sort(key=lambda x: x['confidence'], reverse=True)
        
        # Get the top prediction
        top_prediction = predictions[0]
        top_label = top_prediction['class']
        top_confidence = top_prediction['confidence']
        
        # Return the prediction result
        return top_label.capitalize(), top_confidence
    else:
        return None, None

# Route for the main page
@app.route('/')
def index():
    return render_template('index.html')

import base64

# Route for uploading an image
@app.route('/upload_image', methods=['POST'])
def upload_image():
    if request.method == 'POST':
        # Check if the file exists in the request
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'})

        # Get the uploaded image file
        file = request.files['file']
        if file:
            # Read the uploaded image
            nparr = np.frombuffer(file.read(), np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            # Convert the image to grayscale for inference
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Perform inference and get results
            label, confidence = perform_inference(gray_image)

            # Convert the image to base64
            _, buffer = cv2.imencode('.jpg', image)
            image_base64 = base64.b64encode(buffer).decode('utf-8')

            # Render HTML template to display image and prediction
            return render_template('image_prediction.html', image_base64=image_base64, prediction=label, confidence=confidence)

    # If no file was uploaded or if the request method is not POST,
    # return an error response
    return jsonify({'error': 'No file uploaded or invalid request method'})


# Route for uploading a video
@app.route('/upload_video', methods=['POST'])
def upload_video():
    if request.method == 'POST':
        # Check if the file exists in the request
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'})

        # Get the uploaded video file
        file = request.files['file']
        if file:
            # Save the uploaded video
            video_path = './static/' + file.filename
            file.save(video_path)

            # Perform deepfake detection on the uploaded video
            prediction_result, confidence, image_path = predict_deepfake(video_path)

            # Render the template with the prediction result
            return render_template('results.html', prediction=prediction_result, confidence=confidence, image_path=image_path)

    # If no file was uploaded or if the request method is not POST,
    # return an error response
    return jsonify({'error': 'No file uploaded or invalid request method'})

@app.route('/get_result_image')
def get_result_image():
    return send_file('result.jpg', mimetype='image/jpg')

if __name__ == "__main__":
    app.run(debug=True)
