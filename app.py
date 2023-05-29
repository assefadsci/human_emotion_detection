# import required libraries
from ultralytics import YOLO
import cv2
import streamlit as st
import av
import numpy as np
from PIL import Image
import tempfile
from streamlit_webrtc import webrtc_streamer

# Set up the Streamlit application
st.set_page_config(
    page_title='Human Emotion Detector',
    page_icon=':smiley:',
    layout='wide'
)

st.title('Human Emotion Detector')
st.sidebar.title('Human Emotion Detector')

# Set up the sidebar menu options
menu = st.sidebar.selectbox('Select an option:', ['About', 'Upload Image', 'Upload Video',
                                                  'WebCam'])
# Load the model
@st.cache_resource
def load_model():
    return YOLO('weights/emotion.pt')

model = load_model()


# detect emotion and  draw bounding boxes
def detect_emotion_and_draw(frame):

    classes = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    conf_threshold = 0.3

    # detect emotions
    detections = model(frame, stream=True)

    # get b_box, confidence and class for each detected emotions
    for detection in detections:
        bounding_boxes = detection.boxes
        for box in bounding_boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0]
            cls = int(box.cls[0])

            # draw bounding boxes for objects that exceed the confidence threshold
            if conf > conf_threshold:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

                # Add a label for the detected object
                text_size, _ = cv2.getTextSize(f'{classes[cls]} {conf:.2f}', cv2.FONT_HERSHEY_PLAIN, 1, 2)
                text_width, text_height = text_size

                # add the class label with in the frame
                if y1 - 30 < 0:
                    y1 = 30
                if x1 + text_width + 5 > frame.shape[1]:
                    x1 = frame.shape[1] - text_width - 5
                if y1 - text_height - 10 < 0:
                    y1 = text_height + 10

                cv2.rectangle(frame, (x1, y1 - 30), (x1 + text_width + 5, y1 - 5),
                              (0, 255, 0), -1)
                cv2.putText(frame, f'{classes[cls]} {conf:.2f}', (x1, y1 - 10),
                            cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 2)
    return frame


# Define a callback function to process each frame
def callback(frame):
    resized_frame = frame.to_ndarray(format="bgr24")
    detect_emotion_and_draw(resized_frame)
    return av.VideoFrame.from_ndarray(resized_frame, format="bgr24")

if menu == 'About':
    st.markdown("""
        ### About

        This application utilizes a custom YOLOv8 model to detect 7 human emotions i.e. 
        Angry, Disgust, Fear, Happy, Sad, Surprise, and Neutral from a variety of 
        sources including image, video, and WebCam.

        ### Source Code

        The source code for this application can be found on my GitHub page:

        - [GitHub](https://github.com/assefadsci/emotion-detector.git)

        ### Dataset

        The data set used to train the YOLOv8 model can be accessed here:

        - [Data](https://drive.google.com/file/d/1e8sdk0SJDS4MMLdb3E4YrKyI2cmUbaD-/view?usp=sharing)

        ### Contact

        If you have any questions, comments or feedbacks please feel free to reach out to me:


        - LinkedIn: linkedin.com/in/efrem-assefa-bbb286237
        """)

# process image
elif menu == 'Upload Image':
    # Allow the user to input an image
    uploaded_file = st.sidebar.file_uploader("Choose an Image", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file)
            image_np = np.array(image)
            detected_image_np = image_np.copy()

            # create two columns to display the images before and after detection
            col1, col2 = st.columns(2)
            with col1:
                st.image(image_np, caption="Uploaded Image", use_column_width=True)
            run_detection = st.sidebar.button('Detect')

            if run_detection:
                detect_emotion_and_draw(image_np)
                with col2:
                    st.image(image_np, caption="Detected Image", use_column_width=True)
        except Exception as e:
            st.error('Failed to load image: '+str(e))

# process video
elif menu == "Upload Video":
    uploaded_video = st.sidebar.file_uploader("Upload a video file...", type=['mp4', 'mpv', 'avi'])
    if uploaded_video:
        cam_width, cam_height = st.slider('Frame Width', 320, 1280, 640, 10),\
            st.slider('Frame Height', 240, 960, 480, 10)
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        temp_file.write(uploaded_video.read())

        cap = cv2.VideoCapture(temp_file.name)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_height)

        # Initialize a Streamlit frame
        st_frame = st.empty()
        stop = False

        # Create two columns in the Streamlit sidebar
        col1, col2 = st.columns(2)

        # Start button to begin playing the video
        if col1.button("Start"):

            # Loop through each frame of the video
            while cap.isOpened():
                try:
                    success, frame = cap.read()
                    if not success:
                        print("Can't receive frame (stream end?). Exiting ...")
                        break
                    detect_emotion_and_draw(frame)
                    st_frame.image(frame, channels="BGR", width=cam_width)
                    if stop or cv2.waitKey(0) & 0xff == 27:
                        break
                except Exception as e:
                    print("Error processing frame:", str(e))

        # Stop video if the stop button is clicked
        if col2.button("Stop"):
            stop = True

# process webcam
elif menu == 'WebCam':

    # Add sliders to adjust the frame width and height
    frame_width = st.slider('Frame Width', 320, 1280, 640, 10)
    frame_height = st.slider('Frame Height', 240, 960, 480, 10)

    # Stream the video using WebRTC
    webrtc_streamer(
        key="human_emotion_detection",
        video_frame_callback=callback,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={"video": True, "audio": False}

    )


