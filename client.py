import socket
import speech_recognition as sr

recognizer = sr.Recognizer()

# Create a socket client
server_ip = '192.168.134.253'
server_port = 12345

client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect((server_ip, server_port))

while True:
    signal = client_socket.recv(1024)
    if signal == b"SensorTapped":
        print("Sensor tapped, performing speech recognition...")
        with sr.Microphone() as source:
            print("Say something:")
            audio = recognizer.listen(source)
        try:
            text = recognizer.recognize_google(audio)
            print("You said:", text)
            client_socket.send(text.encode()) #Send back as text
        except sr.UnknownValueError:
            print("Speech recognition could not understand audio")
        except sr.RequestError as e:
            print(f"Could not request results from Google Speech Recognition service; {e}")
