import streamlit as st
import cv2
import numpy as np
import tempfile
from inference_sdk import InferenceHTTPClient
from deepfake_detection import predict_deepfake

# Set theme to dark mode
st.markdown(
"""
<style>
body {
    color: white;
    background-color: #121212;
}
</style>
""",
unsafe_allow_html=True
)

# Initialize the InferenceHTTPClient
CLIENT = InferenceHTTPClient(
    api_url="https://classify.roboflow.com",
    api_key="8RSJzoEweFB7NxxNK6fg"
)

# Function to perform face anti-spoofing inference and display results
def perform_face_verification(image):
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
        st.markdown(f"### Face Verification Result: {top_label.capitalize()} ({top_confidence:.2f} confidence)")
        
        # Display verification status based on prediction
        if top_label == 'real':
            st.success("✅ Verified!")
            return True
        else:
            st.error("❌ Spoof Detected!")
            return False
    else:
        st.write("No predictions found.")
        return False

# Function to perform deepfake detection and display results
def perform_deepfake_detection(video_path):
    # Perform deepfake detection on the input video
    prediction_result, confidence, image_path = predict_deepfake(video_path)

    # Render the template with the prediction result
    st.markdown(f"### Deepfake Detection Result: {prediction_result.capitalize()} ({confidence:.2f} confidence)")
    if prediction_result.lower() == 'fake':
        st.error("❌ Deepfake Detected!")
    else:
        st.success("✅ No Deepfake Detected!")
    st.video(video_path, format='video/mp4')

# Streamlit app code
def main():
    st.title('Secure Vision')
    
    # Description of the app
    st.markdown("""
    This is a face verification and deepfake detection app.
    You can first verify the authenticity of a face, and then check if a video contains a deepfake.
    """)

    # Face verification task
    st.sidebar.markdown("## Face Verification")
    
    # Add a button to capture an image for face verification
    if st.sidebar.button("Capture Image for Face Verification 📸"):
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

        # Perform face verification and move to deepfake detection if real
        if perform_face_verification(gray_frame):
            st.session_state.face_verification_done = True

    # Deepfake detection task
    if st.session_state.get('face_verification_done', False):
        st.sidebar.markdown("## Deepfake Detection")
        
        # Upload video for deepfake detection
        uploaded_video = st.sidebar.file_uploader("Upload Video for Deepfake Detection", type=["mp4"])
        if uploaded_video is not None:
            # Save the uploaded video
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                temp_file.write(uploaded_video.read())
                video_path = temp_file.name

            # Perform deepfake detection and display results
            perform_deepfake_detection(video_path)

if __name__ == "__main__":
    main()
