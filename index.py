#myenv combined command
#conda activate myenv && python -m venv myenv && source myenv/bin/activate

from ultralytics import YOLO
import pyttsx3
import requests
import speech_recognition as sr
import cv2
import threading
import math 
import socket
import requests
import sched
import face_recognition
from deepface import DeepFace
from datetime import datetime, timedelta
#Version 4

import json
import io
import time
import os
import random
from PIL import Image
import base64
import re
import pytesseract
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import torch

from model import Net
from utils import ConfigS, ConfigL, download_weights

last_recognition_time = {}

def send_text_to_pi(text):
    url = f'http://192.168.134.253:5500/text_response'
    json_data=json.dumps({'text':text})
    headers = {'Content-Type': 'application/json'}
    response = requests.post(url, data=json_data, headers=headers)
    if response.status_code == 200:
        print("Text sent successfully.")
    else:
        print(f"Error: {response.status_code}")
        print(response.text)

def verify_faces_from_video_stream(video_url="http://192.168.134.253:5500/video_feed"):
    known_people_folder = "Faces/"
    last_recognition_time = {}  # Dictionary to track the time of last recognition for each person
    no_face_recognition_start_time = None  # Track the start time when no face is recognized

    # Function to load and preprocess all reference images for a person
    def load_reference_images(person_folder):
        reference_images = []
        for filename in os.listdir(person_folder):
            if filename.endswith(".jpg") or filename.endswith(".jpeg"):
                image_path = os.path.join(person_folder, filename)
                reference_images.append(face_recognition.load_image_file(image_path))
        return reference_images

    # Load reference images for all known people
    known_people = {}
    for person_folder in os.listdir(known_people_folder):
        person_folder_path = os.path.join(known_people_folder, person_folder)
        if os.path.isdir(person_folder_path):
            known_people[person_folder] = load_reference_images(person_folder_path)

    # Initialize the video stream
    while True:
        response = requests.get(video_url, stream=True)
        if response.status_code == 200:
            bytes_data = bytes()
            for chunk in response.iter_content(chunk_size=1024):
                bytes_data += chunk
                a = bytes_data.find(b'\xff\xd8')  # Start marker of the frame
                b = bytes_data.find(b'\xff\xd9')  # End marker of the frame
                if a != -1 and b != -1:
                    jpg = bytes_data[a:b+2]  # Extract the frame
                    bytes_data = bytes_data[b+2:]  # Remove the extracted frame from bytes_data
                    frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)

                    # Resize the frame to a smaller size for faster processing
                    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

                    # Convert the frame to RGB (face_recognition requires RGB images)
                    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

                    # Find face locations in the frame
                    face_locations = face_recognition.face_locations(rgb_small_frame)

                    if len(face_locations) > 0:
                        # Reset the timer if a face is recognized
                        no_face_recognition_start_time = None

                        # Verify the input frame against reference images for all known people
                        is_verified = False
                        matched_person = None
                        for person_name, reference_images in known_people.items():
                            for reference_image in reference_images:
                                try:
                                    small_face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
                                    reference_face_encodings = face_recognition.face_encodings(reference_image)
                                    if small_face_encodings and reference_face_encodings:
                                        result = face_recognition.compare_faces(reference_face_encodings, small_face_encodings[0])
                                    else:
                                        continue  # Skip verification if no face encodings are available
                                except ValueError as e:
                                    continue  # Skip verification if an error occurs
                                if result and result[0]:
                                    now = datetime.now()
                                    last_recognition = last_recognition_time.get(person_name, datetime.min)
                                    time_difference = now - last_recognition
                                    if time_difference >= timedelta(minutes=15):
                                        is_verified = True
                                        matched_person = person_name
                                        last_recognition_time[person_name] = now
                                        break  # Exit the loop if a match is found
                            if is_verified:
                                break  # Exit the outer loop if a match is found
                        # Display the verification result on the frame
                        if is_verified:
                            cv2.putText(frame, f"Matched: {matched_person}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                            send_text_to_pi("I found "+matched_person)
                            return
                        else:
                            cv2.putText(frame, "No match found", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    else:
                        # Start the timer if no face is recognized
                        if no_face_recognition_start_time is None:
                            no_face_recognition_start_time = datetime.now()
                        else:
                            # Check if 10 seconds have passed with no face recognition
                            time_difference = datetime.now() - no_face_recognition_start_time
                            if time_difference >= timedelta(seconds=10):
                                return  # Return if 10 seconds have passed with no face recognition

                    # Display the frame with the verification result
                    # cv2.imshow('Face Verification', frame)

        # Exit the loop if the 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()


def detect_text_with_google_vision(image_path):
    api_key = "AIzaSyBQJdOXIQAGL50jLSx6-zFeD2ybAWCch9E"
    # Prepare the image data by encoding it in base64
    with open(image_path, "rb") as image_file:
        image_data = base64.b64encode(image_file.read()).decode('utf-8')

    # Prepare the API request
    vision_url = "https://vision.googleapis.com/v1/images:annotate?key=" + api_key
    headers = {"Content-Type": "application/json"}

    # Create a JSON request with the base64-encoded image data and specify OCR
    request_data = {
        "requests": [
            {
                "image": {"content": image_data},
                "features": [{"type": "TEXT_DETECTION"}]
            }
        ]
    }

    # Make the API request
    response = requests.post(vision_url, headers=headers, json=request_data)

    # Check the response and extract the text
    if response.status_code == 200:
        results = response.json()
        if "textAnnotations" in results["responses"][0]:
            detected_text = results["responses"][0]["textAnnotations"][0]["description"]
            return detected_text
        else:
            return "No text detected in the image."
    else:
        return "Error: " + response.text

def easy_ocr_try(image_path):
    try:
        # Initialize the EasyOCR reader
        reader = easyocr.Reader(['en'])  # Specify the desired language(s)

        # Perform OCR on the image
        result = reader.readtext(image_path)

        # Extract and return the recognized text
        recognized_text = [detection[1] for detection in result]
        return ' '.join(recognized_text)

    except Exception as e:
        return str(e)

def check_touch_pi(glob_val):
    print("Hello World ",glob_val)
    return glob_val+1

def capture_and_save_frame(output_path):
    video_url = "http://192.168.134.253:5500/video_feed"
    response = requests.get(video_url, stream=True)
    if response.status_code == 200:
        # Create a VideoCapture object from the stream content
        bytes_data = bytes()
        for chunk in response.iter_content(chunk_size=1024):
            bytes_data += chunk
            a = bytes_data.find(b'\xff\xd8')  # Find the start of the JPEG frame
            b = bytes_data.find(b'\xff\xd9')  # Find the end of the JPEG frame

            if a != -1 and b != -1:
                jpg = bytes_data[a:b + 2]  # Extract the JPEG frame
                bytes_data = bytes_data[b + 2:]  # Remove the processed data

                # Convert the JPEG frame to a NumPy array
                frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)

                # Check if the frame was successfully decoded
                if frame is not None:
                    # Save the frame as an image
                    cv2.imwrite(output_path, frame)
                    print(f"Frame saved as {output_path}")
                    break  # Exit the loop after saving the frame
                else:
                    print("Error: Could not decode frame.")
        else:
            print("Error: Could not find a complete frame in the stream.")
    else:
        print("Error: Could not access the video feed.")

    # Close the response object
    response.close()
    return ("DONE")

# def capture_and_save_frame_pi(output_folder, num_frames=100):
#     # Create the output folder if it doesn't exist
#     os.makedirs(output_folder, exist_ok=True)

#     video_capture = cv2.VideoCapture("http://192.168.134.253:5500/video_feed")
    
#     frame_count = 0

#     while frame_count < num_frames:
#         ret, frame = video_capture.read()
        
#         if not ret:
#             print("Error: Could not read frame.")
#             break

#         frame_count += 1
#         frame_filename = os.path.join(output_folder, f"{frame_count}.jpg")
#         cv2.imwrite(frame_filename, frame)
#         print(f"Frame {frame_count} saved as {frame_filename}")

#     print(f"Captured {frame_count} frames.")
#     video_capture.release()
#     cv2.destroyAllWindows()


def recognize_first_person():
    # Path to the folder containing known faces
    known_faces_folder = "Faces/"

    # Load known faces
    known_face_encodings = []
    known_face_names = []

    for filename in os.listdir(known_faces_folder):
        if filename.endswith(".jpg"):
            name = os.path.splitext(filename)[0]
            image = face_recognition.load_image_file(os.path.join(known_faces_folder, filename))
            face_encoding = face_recognition.face_encodings(image)[0]
            known_face_encodings.append(face_encoding)
            known_face_names.append(name)

    # Initialize the webcam (you can also use cv2.VideoCapture(0) for a connected camera)
    video_capture = cv2.VideoCapture(0)

    # Initialize Haar Cascade for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    recognized_name = "Unknown"

    while True:
        # Capture a single frame from the webcam
        ret, frame = video_capture.read()

        # Convert the frame to grayscale for Haar Cascade face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the grayscale frame
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Iterate over detected faces
        for (x, y, w, h) in faces:
            # Crop the face from the frame
            face_image = frame[y:y + h, x:x + w]

            # Convert the cropped face image to RGB (required for face_recognition)
            rgb_face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)

            # Recognize the face
            face_encodings = face_recognition.face_encodings(rgb_face_image)

            if len(face_encodings) > 0:
                # Compare the detected face with known faces
                matches = face_recognition.compare_faces(known_face_encodings, face_encodings[0])

                if True in matches:
                    first_match_index = matches.index(True)
                    recognized_name = known_face_names[first_match_index]

                # Release the webcam and close the OpenCV window
                video_capture.release()
                cv2.destroyAllWindows()

                return recognized_name

        # Display the resulting frame
        cv2.imshow('Video', frame)

        # Exit the loop when the 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close the OpenCV window
    video_capture.release()
    cv2.destroyAllWindows()




def capture_and_save_frame_pi(output_folder, num_frames=100):
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Load a pre-trained face detection model (e.g., Haar Cascade or Dlib)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    video_capture = cv2.VideoCapture("http://192.168.134.253:5500/video_feed")
    
    frame_count = 0

    while frame_count < num_frames:
        ret, frame = video_capture.read()
        
        if not ret:
            print("Error: Could not read frame.")
            break

        # Detect faces in the frame
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        if len(faces) > 0:
            # Assume there's only one face in the frame for simplicity
            x, y, w, h = faces[0]

            # Crop and save only the detected face
            face = frame[y:y+h, x:x+w]
            frame_count += 1
            frame_filename = os.path.join(output_folder, f"{frame_count}.jpg")
            cv2.imwrite(frame_filename, face)
            print(f"Face {frame_count} saved as {frame_filename}")

    print(f"Captured {frame_count} faces.")
    video_capture.release()
    cv2.destroyAllWindows()






def capture_images_to_folder(arg_name):
    output_folder="Faces/"+arg_name
    num_images_to_capture=100
    # capture_interval_ms=1000
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Initialize the webcam

    image_count = 0

    while image_count < num_images_to_capture:
        image_count += 1
        capture_and_save_frame("Faces/"+arg_name+"/"+str(image_count)+".jpg")

        # Increment the image count
        

        # Save the captured frame as an image

        # time.sleep(0.4)

    # Release the webcam and close the OpenCV window
    cv2.destroyAllWindows()

def read_text(image_path):
    
    image = Image.open(image_path)
    text = pytesseract.image_to_string(image)
    cleaned_string = re.sub(r"[':;_`~-—]", '', text)
    cleaned_string2 = re.sub(r'["]', '', cleaned_string)
    normalized_string = ' '.join(cleaned_string2.split())
    return normalized_string 
    
def image_captioning_inference(img_path, checkpoint_name='model.pt', size='L', temperature=1.0):
    config = ConfigL() if size.upper() == 'L' else ConfigS()

    # set seed
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)
    torch.backends.cudnn.deterministic = True

    is_cuda = torch.cuda.is_available()
    device = 'cuda' if is_cuda else 'cpu'

    ckp_path = os.path.join(config.weights_dir, checkpoint_name)

    assert os.path.isfile(img_path), 'Image does not exist'

    img = Image.open(img_path)

    model = Net(
        clip_model=config.clip_model,
        text_model=config.text_model,
        ep_len=config.ep_len,
        num_layers=config.num_layers,
        n_heads=config.n_heads,
        forward_expansion=config.forward_expansion,
        dropout=config.dropout,
        max_len=config.max_len,
        device=device
    )

    if not os.path.exists(config.weights_dir):
        os.makedirs(config.weights_dir)

    if not os.path.isfile(ckp_path):
        download_weights(ckp_path, size)

    checkpoint = torch.load(ckp_path, map_location=device)
    model.load_state_dict(checkpoint, strict=False)
    model.eval()

    with torch.no_grad():
        caption, _ = model(img, temperature)

    return caption

def live_object_detection(detect_obj):
    start_time = time.time()
    #line above
    # cap = cv2.VideoCapture(0)
    #above 2
    camera_url = "http://192.168.134.253:5500/video_feed"
    cap = cv2.VideoCapture(camera_url)
    cap.set(3, 640)
    cap.set(4, 480)


    model = YOLO("yolo-Weights/yolov8n.pt")


    classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
                  "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
                  "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
                  "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
                  "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
                  "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
                  "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
                  "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
                  "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
                  "teddy bear", "hair drier", "toothbrush"
                  ]


    while True:
        if time.time() - start_time > 10:
            return "object not found"
        #above two lines
        success, img = cap.read()
        results = model(img, stream=True)


        for r in results:
            boxes = r.boxes

            for box in boxes:

                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) 


                #cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 2) #This was for creating a box around all detected objects

                confidence = math.ceil((box.conf[0]*100))/100

                cls = int(box.cls[0])

                org = [x1, y1]
                font = cv2.FONT_HERSHEY_SIMPLEX
                fontScale = 1
                color = (255, 0, 0)
                thickness = 2
                
                if(classNames[cls]==detect_obj and confidence>=0.3):
                        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 2)
                        cv2.putText(img, classNames[cls], org, font, fontScale, (255, 255, 0), thickness)
                        
                        # print(' width is ',x2-x1)
                        # print (' height is ',y2-y1)
                        
                        print(x1," ",x2," ",y1," ",y2)
                        
                        detected_object_img = img[y1:y2, x1:x2]
                        cv2.imwrite("detected_frame.jpg", img)
                        cv2.imwrite("detected_object.jpg", detected_object_img)
                        cap.release()
                        cv2.destroyAllWindows()
                        
                        if(y1>=240 and y2>=240):
                            if(x2<=320):
                                found_element=detect_obj+" found on bottom right"
                                return found_element
                            elif(x2>=320 and x1<=320):
                                found_element=detect_obj+" found on bottom center"
                                return found_element
                            else:
                                found_element=detect_obj+" found on bottom left"
                                return found_element
                        elif(y1<=240 and y2>=240):
                            if(x2<=320):
                                found_element=detect_obj+" found on middle right"
                                return found_element
                            elif(x2>=320 and x1<=320):
                                found_element=detect_obj+" found on middle center"
                                return found_element
                            else:
                                found_element=detect_obj+" found on middle left"
                                return found_element
                        else:
                            if(x2<=320):
                                found_element=detect_obj+" found on top right"
                                return found_element
                            elif(x2>=320 and x1<=320):
                                found_element=detect_obj+" found on top center"
                                return found_element
                            else:
                                found_element=detect_obj+" found on top left"
                                return found_element
                        
def recognize_speech():
    # Initialize the recognizer
    recognizer = sr.Recognizer()

    # Initialize the text-to-speech engine
    engine = pyttsx3.init()

    # Capture audio from the microphone
    with sr.Microphone() as source:
        print("Listening... ")
        
        send_text_to_pi("Listening")
        recognizer.adjust_for_ambient_noise(source)  # Adjust for ambient noise
        audio = recognizer.listen(source)

    # Recognize the speech using Google Web Speech API
    
    try:
        text = recognizer.recognize_google(audio)
        return text  # Return the recognized text
    except sr.UnknownValueError:
        send_text_to_pi("Sorry, I could not understand the audio.")
        return "Sorry, I could not understand the audio."
    except sr.RequestError as e:
        send_text_to_pi("Sorry, an error occurred. Could not request results; {0}".format(e))
        return ("Sorry, an error occurred. Could not request results; {0}".format(e))

def speak_text(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()
    print(text)
#print(live_object_detection("cell phone"))
# print(image_captioning_inference("detected_frame.jpg"))
#print(read_text("images/image.jpeg"))

def run():
    recognized_text = recognize_speech()

    res = recognized_text.split()
    print(recognized_text)
    print("Run command : ",res[0])
    if(res[0]=="read"):
        capture_and_save_frame("images/read_image.jpg")
        read_text_img = detect_text_with_google_vision("images/read_image.jpg")
        #read_text_img=(read_text("images/read_image.jpg"))
        if read_text_img:
            #speak_text(read_text)
            print(read_text_img)
            send_text_to_pi(read_text_img)
        else:
            send_text_to_pi("Text not recognizable, please try again in better lighting")
    elif(res[0]=="analyse"):
        capture_and_save_frame("images/analyse_pic.jpg")
        analyse_pic=(image_captioning_inference("images/analyse_pic.jpg"))
        send_text_to_pi(analyse_pic)
        # speak_text(analyse_pic)
    elif(res[0]=='train'):
        send_text_to_pi("Face training started")
        capture_and_save_frame_pi("Faces/"+res[1])
        send_text_to_pi("Image Training Completed")
    elif(res[0]=='recognise'):
        detected_person=recognize_first_person()
        send_text_to_pi(detected_person)
    elif(res[0]=='find'):
        classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
                      "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
                      "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
                      "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
                      "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
                      "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
                      "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
                      "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
                      "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
                      "teddy bear", "hair drier", "toothbrush"
                      ]


        if len(res)>=3:
            print(len(res))
            combo_text=res[1]+" "+res[2]
            print(res[1]," and ",combo_text)
        if res[1] in classNames:
            key=res[1]
            print("Key object is : ",key)
            find_text=live_object_detection(key)
            # speak_text(find_text)
            send_text_to_pi(find_text)


        elif ((len(res)>=3)):
            if(combo_text in classNames):
                key=combo_text
                print("Key object is : ",key)
                find_text=live_object_detection(key)
                # speak_text(find_text)
                send_text_to_pi(find_text)
            else:
                # speak_text("Invalid Object mentioned")
                send_text_to_pi("Invalid Object mentioned")
        else:
            # speak_text("Invalid Object mentioned")
            send_text_to_pi("Invalid Object mentioned")
    elif(res[0]=="search"):
        # speak_text("Searching on the internet")
        send_text_to_pi("Searching on the internet")

if __name__ == "__main__":
    # run()
    
    #auto run
    
    while True:
        url = f'http://192.168.134.253:5500/get_touch'
        response = requests.get(url)
        if(response.text=="YES"):
                run()
        else:
            time.sleep(1.0)
    
    #detected_text = detect_text_with_google_vision()

        

#print(recognized_text)
                        






