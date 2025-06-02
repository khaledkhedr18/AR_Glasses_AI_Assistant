import socket
import json
import base64
import time
import datetime
from PyQt5.QtCore import QObject

class Network_Handler(QObject):
    """
    Handler for network communications including server connections
    and data transmission for online services.
    """

    def __init__(self, server_ip='192.168.1.65', server_port=4040):
        """
        Initialize the Network_Handler.

        Args:
            server_ip (str): IP address of the translation server
            server_port (int): Port number of the translation server
        """
        super().__init__()
        self.SERVER_IP = server_ip
        self.SERVER_PORT = server_port
        self.sock = None
        self.file_obj = None
        self.connected = False

    def check_internet_connection(self, timeout=2):
        """
        Check if the device is connected to the internet.

        Args:
            timeout (float): Connection timeout in seconds

        Returns:
            bool: True if connected, False otherwise
        """
        try:
            sock = socket.create_connection(("www.google.com", 80), timeout=timeout)
            sock.close()
            return True
        except (socket.error, OSError):
            return False

    def connect_to_server(self, max_retries=2):
        """
        Connect to the translation server.

        Args:
            max_retries (int): Maximum number of connection attempts

        Returns:
            bool: True if connection successful, False otherwise
        """
        for attempt in range(max_retries):
            try:
                print(f"Connecting to {self.SERVER_IP}:{self.SERVER_PORT}...")
                self.sock = socket.create_connection((self.SERVER_IP, self.SERVER_PORT), timeout=5)
                self.file_obj = self.sock.makefile('r')
                self.connected = True
                print("Connection established")
                return True
            except (ConnectionRefusedError, OSError) as e:
                print(f"Connection error: {e}. Retrying...")
                time.sleep(1)

        print("Failed to connect to server")
        return False

    def disconnect_from_server(self):
        """Close the connection to the translation server."""
        if self.sock:
            try:
                self.sock.close()
            except Exception:
                pass
        self.connected = False
        self.file_obj = None
        self.sock = None

    def send_and_receive(self, data_type, source_lang, target_lang, image_data=None, text=None):
        """
        Send a request to the server and receive the response.

        Args:
            data_type (str): Type of data being sent ("text", "image", or "text_and_image")
            source_lang (str): Source language code
            target_lang (str): Target language code
            image_data (str, optional): Base64 encoded image data or image path
            text (str, optional): Text to translate

        Returns:
            str or None: Server response if successful, None otherwise
        """
        if not self.connected:
            if not self.connect_to_server():
                return None

        try:
            success = False
            if data_type == "image" and image_data:
                success = self.send_image(image_data, source_lang, target_lang)
            elif data_type == "text" and text:
                success = self.send_text(text, source_lang, target_lang)
            elif data_type == "text_and_image" and image_data:
                success = self.send_combined(text or "", image_data, source_lang, target_lang)

            if not success:
                return None

            response = self.receive_response()
            return response

        except Exception as e:
            print(f"Communication error: {e}")
            self.disconnect_from_server()
            return None

    def send_text(self, text, source_lang, target_lang):
        """
        Send text to the server for translation.

        Args:
            text (str): Text to translate
            source_lang (str): Source language code
            target_lang (str): Target language code

        Returns:
            bool: True if sending successful, False otherwise
        """
        if not self.connected:
            return False

        try:
            message = {
                "type": "text",
                "data": text,
                "source_lang": source_lang,
                "target_lang": target_lang,
                "timestamp": datetime.datetime.now().isoformat()
            }
            data = json.dumps(message) + "\n"
            self.sock.sendall(data.encode('utf-8'))
            return True
        except Exception as e:
            print(f"Error sending text: {e}")
            self.connected = False
            return False

    def send_image(self, image_path, source_lang, target_lang):
        """
        Send an image to the server for OCR and translation.

        Args:
            image_path (str): Path to the image file
            source_lang (str): Source language code
            target_lang (str): Target language code

        Returns:
            bool: True if sending successful, False otherwise
        """
        if not self.connected:
            return False

        try:
            with open(image_path, "rb") as image_file:
                image_bytes = image_file.read()

            base64_image = base64.b64encode(image_bytes).decode('utf-8')
            message = {
                "type": "image",
                "data": base64_image,
                "source_lang": source_lang,
                "target_lang": target_lang,
                "timestamp": datetime.datetime.now().isoformat()
            }
            data = json.dumps(message) + "\n"
            self.sock.sendall(data.encode('utf-8'))
            return True
        except Exception as e:
            print(f"Error sending image: {e}")
            self.connected = False
            return False

    def send_combined(self, text, image_data, source_lang, target_lang):
        """
        Send both text and image data to the server.

        Args:
            text (str): Optional text prompt
            image_data (str): Base64 encoded image data
            source_lang (str): Source language code
            target_lang (str): Target language code

        Returns:
            bool: True if sending successful, False otherwise
        """
        if not self.connected:
            return False

        try:
            message = {
                "type": "text_and_image",
                "text": text if text else "",
                "image": image_data,
                "source_lang": source_lang,
                "target_lang": target_lang,
                "timestamp": datetime.datetime.now().isoformat()
            }
            data = json.dumps(message) + "\n"
            self.sock.sendall(data.encode('utf-8'))
            return True
        except Exception as e:
            print(f"Error sending combined data: {e}")
            return False

    def receive_response(self, timeout=10):
        """
        Receive a response from the server.

        Args:
            timeout (int): Maximum wait time in seconds

        Returns:
            str or None: Server response if received, None otherwise
        """
        import select

        self.sock.setblocking(0)
        ready = select.select([self.sock], [], [], timeout)
        if ready[0]:
            try:
                data = self.sock.recv(4096).decode('utf-8')
                if data:
                    try:
                        message = json.loads(data)
                        return message.get("data")
                    except json.JSONDecodeError:
                        print("Received invalid JSON.")
                        return None
                else:
                    print("No data received.")
                    return None
            except socket.error as e:
                print(f"Socket error: {e}")
                return None
        else:
            print("No response received within timeout period.")
            return None

    def cleanup(self):
        """Clean up network resources."""
        self.disconnect_from_server()
