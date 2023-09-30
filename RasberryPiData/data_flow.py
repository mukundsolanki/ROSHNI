from gpiozero import Button
import socket
import pyttsx3

touch_sensor = Button(17)

# Create a socket server
server_ip = '192.168.134.253'
server_port = 12345

server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind((server_ip, server_port))
server_socket.listen(1)

print(f"Server listening on {server_ip}:{server_port}")

engine = pyttsx3.init()

while True:
    client_socket, client_address = server_socket.accept()
    print(f"Connection from {client_address}")

    try:
        while True:
            touch_sensor.wait_for_release()
            touch_sensor.wait_for_press()
            client_socket.send(b"SensorTapped")
            print("Sensor tapped")
            recognized_text = client_socket.recv(1024).decode()
            print("Received text:", recognized_text)

            engine.say(recognized_text)
            engine.runAndWait()
    finally:
        client_socket.close()
