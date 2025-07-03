import socket
import os
import json
import base64
import time
import datetime
from utils.Config import NETWORK_CONFIG
from utils.Logging import Logger

class SocketHandler:
    """
    Handler for network communications including server connections
    and data transmission for online services.
    """

    def __init__(self, server_ip=None, server_port=None):
        """
        Initialize the Network_Handler.

        Args:
            server_ip (str, optional): IP address of the translation server
            server_port (int, optional): Port number of the translation server
        """
        self.logger = Logger()

        # Use provided values or fall back to config
        self.SERVER_IP = server_ip or NETWORK_CONFIG.get('SERVER_IP', '192.168.1.65')
        self.SERVER_PORT = server_port or NETWORK_CONFIG.get('SERVER_PORT', 4040)
        self.SOCKET_TIMEOUT = NETWORK_CONFIG.get('SOCKET_TIMEOUT', 5)
        self.MAX_RETRIES = NETWORK_CONFIG.get('MAX_RETRIES', 3)
        self.RETRY_DELAY = NETWORK_CONFIG.get('RETRY_DELAY', 1)
        self.REQUEST_TIMEOUT = NETWORK_CONFIG.get('REQUEST_TIMEOUT', 30)
        self.RESPONSE_TIMEOUT = NETWORK_CONFIG.get('RESPONSE_TIMEOUT', 30)
        self.CHUNK_SIZE = NETWORK_CONFIG.get('CHUNK_SIZE', 4096)

        self.sock = None
        self.file_obj = None
        self.connected = False

        self.logger.info(f"Network_Handler initialized with server {self.SERVER_IP}:{self.SERVER_PORT}")

    def check_internet_connection(self, timeout=None):
        """
        Check if the device has an active internet connection.
        More robust implementation that tries multiple methods.
        """
        if timeout is None:
            timeout = NETWORK_CONFIG.get('STATUS_CHECK_TIMEOUT', 2.0)

        # Try multiple methods to confirm internet connectivity
        test_urls = [
            'www.google.com',  # Google
            'www.cloudflare.com',  # Cloudflare
            '1.1.1.1'  # Cloudflare DNS
        ]
        test_port = 80

        for url in test_urls:
            try:
                # Try DNS resolution first
                socket.getaddrinfo(url, test_port)

                # Then try actual connection
                sock = socket.create_connection((url, test_port), timeout=timeout)
                sock.close()
                self.logger.info(f"Internet connection confirmed via {url}")
                return True
            except (socket.gaierror, socket.timeout, ConnectionRefusedError) as e:
                self.logger.debug(f"Connection test failed for {url}: {str(e)}")
                continue
            except Exception as e:
                self.logger.debug(f"Unexpected error testing {url}: {str(e)}")
                continue

        self.logger.warning("All internet connection tests failed")
        return False

    def connect_to_server(self, max_retries=None):
        """
        Connect to the translation server.

        Args:
            max_retries (int, optional): Maximum number of connection attempts

        Returns:
            bool: True if connection successful, False otherwise
        """
        if max_retries is None:
            max_retries = self.MAX_RETRIES

        for attempt in range(max_retries):
            try:
                self.logger.info(f"Connecting to {self.SERVER_IP}:{self.SERVER_PORT} (attempt {attempt+1}/{max_retries})...")
                self.sock = socket.create_connection((self.SERVER_IP, self.SERVER_PORT),
                                                    timeout=self.SOCKET_TIMEOUT)
                self.file_obj = self.sock.makefile('r')
                self.connected = True
                self.logger.info("Connection established")
                return True
            except (ConnectionRefusedError, OSError) as e:
                self.logger.warning(f"Connection error: {e}. Retrying...")
                time.sleep(self.RETRY_DELAY)

        self.logger.error("Failed to connect to server")
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

    def send_and_receive(self, data_type, source_lang, target_lang, image_data=None, text=None, audio_data=None):
        """
        Send a request to the server and receive the response.
        Now supports audio data types.
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
            elif data_type == "audio" and audio_data:
                success = self.send_audio(audio_data, source_lang, target_lang)
            elif data_type == "audio_and_image" and audio_data and image_data:
                success = self.send_combined_audio_image(audio_data, image_data, source_lang, target_lang)
            elif data_type == "text_and_image" and image_data:
                success = self.send_combined(text or "", image_data, source_lang, target_lang)

            if not success:
                return None

            response = self.receive_response()
            return response

        except Exception as e:
            self.logger.error(f"Communication error: {e}")
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
            self.logger.error(f"Error sending text: {e}")
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
            self.logger.error(f"Error sending image: {e}")
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
            self.logger.error(f"Error sending combined data: {e}")
            return False

    def send_audio(self, audio_path, source_lang, target_lang):
        """Send audio file to server"""
        if not self.connected:
            return False

        try:
            if not os.path.exists(audio_path):
                self.logger.error(f"Audio file not found: {audio_path}")
                return False
            with open(audio_path, "rb") as audio_file:
                audio_bytes = audio_file.read()

            base64_audio = base64.b64encode(audio_bytes).decode('utf-8')
            message = {
                "type": "audio",
                "data": base64_audio,
                "source_lang": source_lang,
                "target_lang": target_lang,
                "timestamp": datetime.datetime.now().isoformat()
            }
            data = json.dumps(message) + "\n"
            self.sock.sendall(data.encode('utf-8'))
            return True
        except Exception as e:
            self.logger.error(f"Error sending audio: {e}")
            self.connected = False
            return False

    def send_combined_audio_image(self, audio_path, image_path, source_lang, target_lang):
        """Send both audio and image to server"""
        if not self.connected:
            return False

        try:
            if not os.path.exists(audio_path):
                self.logger.error(f"Audio file not found: {audio_path}")
                return False
            if not os.path.exists(image_path):
                self.logger.error(f"Image file not found: {image_path}")
                return False
            # Read audio file
            with open(audio_path, "rb") as audio_file:
                audio_bytes = audio_file.read()
            base64_audio = base64.b64encode(audio_bytes).decode('utf-8')

            # Read image file
            with open(image_path, "rb") as image_file:
                image_bytes = image_file.read()
            base64_image = base64.b64encode(image_bytes).decode('utf-8')

            message = {
                "type": "audio_and_image",
                "audio": base64_audio,
                "image": base64_image,
                "source_lang": source_lang,
                "target_lang": target_lang,
                "timestamp": datetime.datetime.now().isoformat()
            }
            data = json.dumps(message) + "\n"
            self.sock.sendall(data.encode('utf-8'))
            return True
        except Exception as e:
            self.logger.error(f"Error sending combined audio+image: {e}")
            return False

    def receive_response(self, timeout=None):
        """
        Receive a response from the server.

        Args:
            timeout (int, optional): Maximum wait time in seconds

        Returns:
            str or None: Server response if received, None otherwise
        """
        import select

        if timeout is None:
            timeout = self.RESPONSE_TIMEOUT

        self.sock.setblocking(0)
        ready = select.select([self.sock], [], [], timeout)
        if ready[0]:
            try:
                data = self.sock.recv(self.CHUNK_SIZE).decode('utf-8')
                if data:
                    try:
                        message = json.loads(data)
                        return message.get("data")
                    except json.JSONDecodeError:
                        self.logger.error("Received invalid JSON.")
                        return None
                else:
                    self.logger.warning("No data received.")
                    return None
            except socket.error as e:
                self.logger.error(f"Socket error: {e}")
                return None
        else:
            self.logger.warning(f"No response received within timeout period ({timeout}s).")
            return None

    def cleanup(self):
        """Clean up network resources."""
        self.disconnect_from_server()
