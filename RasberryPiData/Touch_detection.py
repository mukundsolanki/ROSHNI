from gpiozero import Button
from picamera import PiCamera
import socket
import io
import pyttsx3

sensor_pin = 17
sensor_button = Button(sensor_pin)

def on_touch():
	print("Touch Detected")
	capture_and_send_image()
	receive_text_and_convert_to_speech()
	
def capture_and_send_image():
	try:
		server_socket = socket.socket()
		server_socket.bind(('0.0.0.0', 8000))
		server_socket.listen(0)
		print("Server waiting for connections...")
		connection, addr = server_socket.accept()
		
		with PiCamera() as camera:
			camera.resolution = (640,480)
			image_stream = io.BytesIO()
			camera.capture(image_stream, format='jpeg')
			image_stream.seek(0)
			image_data = image_stream.read()
			connection.sendall(image_data)
			
		print("Image sent Sucessfully...")
		connection.close()
		
	except Exception as e:
		print(f"Error: {str(e)}")
		
	finally:
		server_socket.close()
		
def receive_text_and_convert_to_speech():
	try:
		server_socket = socket.socket()
		server_socket.bind(('0.0.0.0', 3000))
		server_socket.listen(0)
		print("Server waiting for text connections....")
		connection, addr = server_socket.accept()
		
		text = connection.recv(1024).decode()
		print("Received text...", text)
		
		convert_text_to_speech(text)
		
	except Exception as e:
		print(f"Error: {str(e)}")
		
	finally:
		connection.close()
		server_socket.close()
		
		
def convert_text_to_speech(text):
	try:
		engine = pyttsx3.init()
		engine.say(text)
		engine.runAndWait()
		print("Text convverted to text sucessfully...")
		
	except Exception as e:
		print("fError converting ttext to speech: {str(e)}")
		
		
sensor_button.when_pressed = on_touch

try:
	print("Waitinf for touch...")
	while True:
		pass
		
except KeyboardInterrupt:
	print("Exiting...")
