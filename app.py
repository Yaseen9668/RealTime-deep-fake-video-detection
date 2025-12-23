import streamlit as st
from inference_sdk import InferenceHTTPClient
import cv2
import tempfile
import os
import numpy as np

# Initialize the InferenceHTTPClient
CLIENT = InferenceHTTPClient(
    api_url="https://classify.roboflow.com",
    api_key="8RSJzoEweFB7NxxNK6fg"
)

# Function to perform inference and display results
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
        
        # Display the prediction result
        st.write(f"### Verification Result: {top_label.capitalize()} ({top_confidence:.2f} confidence)")
        
        # Display verification status based on prediction
        if top_label == 'real':
            st.success("✅ Verified!")
        else:
            st.error("❌ Spoof Detected!")
    else:
        st.write("No predictions found.")

# Streamlit app code
def main():
    st.title('Secure Vision')
    
    # Description of the app
    st.markdown("""
    This is a face verification app that helps in detecting if a face is real or a spoof.
    You can either upload an image or capture one using your webcam.
    """)
    st.markdown("---")
    st.header("Choose Input Method")
    
    # Markdown options for input methods
    input_method = st.radio(
        "",
        ("Upload Image 📤", "Capture Image 📸")
    )

    # Upload image
    if input_method == "Upload Image 📤":
        uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            # Read the uploaded image
            image = np.array(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(image, cv2.IMREAD_COLOR)

            # Convert the image to grayscale for inference
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Perform inference and display results
            perform_inference(gray_image)

            # Display the uploaded image
            st.image(image, channels="BGR", caption='Uploaded Image', use_column_width=True)

    # Capture image
    else:
        # Add a button to capture an image
        if st.button("Capture Image 📸"):
            # Create a VideoCapture object
            cap = cv2.VideoCapture(0)

            # Check if the webcam is opened correctly
            if not cap.isOpened():
                st.error("Error: Could not open the camera.")
                return

            # Read a frame from the camera
            ret, frame = cap.read()

            if not ret:
                st.error("Error: Failed to capture frame from the camera.")
                return

            # Release the VideoCapture object
            cap.release()

            # Convert the frame to grayscale for inference
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Perform inference and display results
            perform_inference(gray_frame)

            # Display the captured image
            st.image(frame, channels="BGR", caption='Captured Image', use_column_width=True)

if __name__ == "__main__":
    main()
