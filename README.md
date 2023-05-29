# Human Emotion Detection

## About

This application utilizes a custom YOLOv8 model to detect 7 human emotions i.e. Angry, Disgust, Fear, Happy, Sad, Surprise, and Neutral from a variety of sources including image, video, and WebCam.

## Libraries used

    - ultralytics: to use the YOLOv8 model
    - cv2: for image manipulation
    - streamlit: to create the web application
    - numpy: for numerical computing
    - PIL: for working with images
    - tempfile: for temporary file storage
    - urllib: for URL handling
    - streamlit_webrtc: for accessing the webcam in the web application
    - av: for video manipulation
    

## How to use the application

The application allows the user to choose between different options for detection:

    - About: Display information about the application and the data set used to train the model.
    - Upload Image: Detect human activities from an image by uploading an image file.
    - Upload Video: Detect human activities from a video by uploading a video file.
    - Use Webcam: Detect human activities using the webcam.
    
The detected activities are shown in bounding boxes and labeled with confidence scores on the image or video. The confidence threshold is set to 0.3. Detected objects that exceed the confidence threshold of 0.3 are labeled with their respective classes. The 7 distinct classes are Angry, Disgust, Fear, Happy, Sad, Surprise, and Neutral.

## Data Set

The data set used to train the YOLOv8 model can be accessed here:

- [Data](https://drive.google.com/file/d/1e8sdk0SJDS4MMLdb3E4YrKyI2cmUbaD-/view?usp=sharing)

## Live Demo

You can try out the application by accessing the live demo hosted on Streamlit:

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://assefadsci-human-emotion-detection-app-9guxuh.streamlit.app/)


## Contact

If you have any questions, comments, or feedbacks, please feel free to reach out to me:

   - [LinkedIn](linkedin.com/in/efrem-assefa-bbb286237)
