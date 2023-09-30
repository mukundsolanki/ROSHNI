import socket

raspberry_pi_ip = '192.168.29.80'
server_port = 8000

text_to_send = "Hello, Raspberry Pi! This is a test message."

def send_text_to_raspberry_pi(text):
    try:
        client_socket = socket.socket()
        client_socket.connect((raspberry_pi_ip, server_port))

        #Send the text
        client_socket.sendall(text.encode())
        print(f"Text sent to Raspberry Pi: {text}")

    except Exception as e:
        print(f"Error sending text to Raspberry Pi: {str(e)}")

    finally:
        client_socket.close()

if __name__ == "__main__":
    send_text_to_raspberry_pi(text_to_send)
