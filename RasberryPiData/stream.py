import io
import time
import pyttsx3
import picamera
import requests
import threading
from flask import Flask, Response, request, jsonify
from gpiozero import Button

touch_sensor = Button(17)
app = Flask(__name__)

pvar = "NO"
count=0 

def speak(text):
    engine=pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

    
def generate_frames():
    while True:
        with picamera.PiCamera() as camera:

            camera.resolution = (640, 480)
            camera.framerate = 30

            time.sleep(2)
            
            stream = io.BytesIO()
            for _ in camera.capture_continuous(stream, format='jpeg', use_video_port=True):
                stream.seek(0)
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + stream.read() + b'\r\n')

                stream.seek(0)
                stream.truncate()
  
  
def onTouch():
    global pvar
    pvar = "YES"

    
touch_sensor.when_pressed = onTouch
@app.route('/get_touch')
def get_touch():
    global pvar
    if(pvar=="YES"):
        pvar="NO"
        return "YES"
    else:
        return "NO"

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/text_response',methods=['POST'])
def text_response():
    text = request.json.get('text')
    if text:
        print(f"Received text: {text}")
        engine=pyttsx3.init()
        engine.say(text)
        engine.runAndWait()
        return jsonify({'message':'Text Received successfully'})
    else:
        return jsonify({'error':'Invalid request'}),400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5500, threaded=True)
