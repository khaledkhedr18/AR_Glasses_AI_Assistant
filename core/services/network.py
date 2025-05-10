import socket
import json
import base64
import datetime
from typing import Optional, Dict, Any
from config.settings import Settings

class NetworkService:
    """Service for handling network communication"""

    def __init__(self):
        self.settings = Settings()
        self.sock = None
        self.file_obj = None
        self.connected = False
        self.max_retries = 3
        self.retry_delay = 2
        self.connect()

    def connect(self) -> bool:
        """Establish connection to the server"""
        retry_count = 0
        while retry_count < self.max_retries:
            try:
                server_ip = self.settings.get("server.ip")
                server_port = self.settings.get("server.port")

                print(f"[{datetime.datetime.now().isoformat()}] Connecting to {server_ip}:{server_port}...")
                self.sock = socket.create_connection((server_ip, server_port))
                self.file_obj = self.sock.makefile('r')
                self.connected = True
                print(f"[{datetime.datetime.now().isoformat()}] Connection established")
                return True
            except (ConnectionRefusedError, OSError) as e:
                retry_count += 1
                print(f"Connection error: {e}. Retry {retry_count}/{self.max_retries} in {self.retry_delay} seconds...")
                import time
                time.sleep(self.retry_delay)
            except Exception as e:
                print(f"Unexpected error: {e}")
                return False
        return False

    def ensure_connection(self) -> bool:
        """Ensure connection is active, reconnect if necessary"""
        if not self.connected or not self.sock:
            return self.connect()
        return True

    def send_message(self, message_type: str, data: Any, source_lang: str, target_lang: str) -> bool:
        """Send a message to the server"""
        if not self.ensure_connection():
            print("Failed to establish connection")
            return False

        try:
            message = {
                "type": message_type,
                "data": data,
                "source_lang": source_lang,
                "target_lang": target_lang,
                "timestamp": datetime.datetime.now().isoformat()
            }

            if message_type == "image":
                message["data"] = base64.b64encode(data).decode('utf-8')

            data = json.dumps(message) + "\n"
            self.sock.sendall(data.encode('utf-8'))
            print(f"[{datetime.datetime.now().isoformat()}] {message_type} sent successfully")
            return True
        except Exception as e:
            print(f"Error sending {message_type}: {e}")
            self.connected = False
            return False

    def receive_response(self, timeout: int = 10) -> Optional[Dict[str, Any]]:
        """Receive a response from the server"""
        if not self.ensure_connection():
            return None

        try:
            self.sock.settimeout(timeout)
            response = self.file_obj.readline()
            if response:
                return json.loads(response)
            return None
        except socket.timeout:
            print("Timeout waiting for response")
            return None
        except json.JSONDecodeError as e:
            print(f"Error decoding response: {e}")
            return None
        except Exception as e:
            print(f"Error receiving response: {e}")
            self.connected = False
            return None

    def cleanup(self):
        """Cleanup network resources"""
        try:
            if self.file_obj:
                self.file_obj.close()
            if self.sock:
                self.sock.close()
            self.connected = False
        except Exception as e:
            print(f"Error during cleanup: {e}")
