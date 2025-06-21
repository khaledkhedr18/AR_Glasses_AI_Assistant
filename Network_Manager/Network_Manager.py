import os
import threading
import time
from utils.logging import Logger
from utils.config import user_config, NETWORK_CONFIG
from Network_Manager.Network_Handlers.Network_Handler import Network_Handler

class NetworkManager:
    """
    The Network Manager class serves as the central hub for handling
    all network-related operations. It provides an interface for the application
    to communicate with network services and manage connectivity.
    """

    def __init__(self):
        """Initialize the Network Manager with required components"""
        self.logger = Logger()
        self.logger.info("Initializing Network Manager")

        # Create Network Handler instance with configuration
        server_ip = NETWORK_CONFIG.get('SERVER_IP')
        server_port = NETWORK_CONFIG.get('SERVER_PORT')
        self.network = Network_Handler(server_ip=server_ip, server_port=server_port)

        # Status tracking
        self.is_connected = False
        self.last_connection_check = 0
        self.connection_check_interval = NETWORK_CONFIG.get('CONNECTION_CHECK_INTERVAL')
        self.status_check_timeout = NETWORK_CONFIG.get('STATUS_CHECK_TIMEOUT')

        # Request settings
        self.max_retries = NETWORK_CONFIG.get('MAX_RETRIES')
        self.thread_join_timeout = NETWORK_CONFIG.get('THREAD_JOIN_TIMEOUT')
        self.verbose_logging = NETWORK_CONFIG.get('VERBOSE_LOGGING', False)

        # Threading resources
        self.net_lock = threading.Lock()
        self.active_threads = []

        # Check initial connection status
        self.update_connection_status()

        if self.verbose_logging:
            self.logger.info(f"Network configuration loaded: Server={server_ip}:{server_port}")

    def update_connection_status(self):
        """
        Update and return the current internet connection status.

        Returns:
            bool: True if connected to internet, False otherwise
        """
        current_time = time.time()

        # Only check connection if enough time has passed since last check
        if current_time - self.last_connection_check > self.connection_check_interval:
            with self.net_lock:
                self.last_connection_check = current_time
                self.is_connected = self.network.check_internet_connection()
                self.logger.info(f"Internet connection status: {'Connected' if self.is_connected else 'Disconnected'}")

        return self.is_connected

    def check_internet_connection(self, timeout=None):
        """
        Check if the device has an active internet connection.

        Args:
            timeout (float, optional): Connection timeout in seconds

        Returns:
            bool: True if connected to internet, False otherwise
        """
        if timeout is None:
            timeout = self.status_check_timeout

        with self.net_lock:
            connected = self.network.check_internet_connection(timeout)
            self.is_connected = connected
            self.last_connection_check = time.time()
            return connected

    def connect_to_server(self, max_retries=None):
        """
        Connect to the remote server.

        Args:
            max_retries (int, optional): Maximum number of connection attempts

        Returns:
            bool: True if connection successful, False otherwise
        """
        if max_retries is None:
            max_retries = self.max_retries

        with self.net_lock:
            if not self.is_connected:
                if not self.check_internet_connection():
                    self.logger.warning("Cannot connect to server: No internet connection")
                    return False

            self.logger.info(f"Connecting to server: {self.network.SERVER_IP}:{self.network.SERVER_PORT}")
            success = self.network.connect_to_server(max_retries)

            if success:
                self.logger.info("Connected to server successfully")
            else:
                self.logger.error("Failed to connect to server")

            return success

    def disconnect_from_server(self):
        """
        Disconnect from the remote server.

        Returns:
            bool: True if disconnection successful or already disconnected
        """
        with self.net_lock:
            if self.network.connected:
                self.logger.info("Disconnecting from server")
                self.network.disconnect_from_server()
                return True
            return True  # Already disconnected

    def send_request(self, data_type, source_lang, target_lang, image_data=None, text=None):
        """
        Send a request to the server and get the response.

        Args:
            data_type (str): Type of data ("text", "image", or "text_and_image")
            source_lang (str): Source language code
            target_lang (str): Target language code
            image_data (str, optional): Path to image file or base64 encoded data
            text (str, optional): Text content

        Returns:
            dict or str or None: Server response or None if failed
        """
        self.logger.info(f"Sending {data_type} request to server")

        with self.net_lock:
            # Ensure we're connected
            if not self.network.connected:
                if not self.connect_to_server():
                    self.logger.error(f"Cannot send {data_type} request: Failed to connect to server")
                    return None

            # Image path validation if provided
            if image_data and data_type in ["image", "text_and_image"] and os.path.exists(image_data):
                self.logger.debug(f"Using image file: {image_data}")
            elif image_data and data_type in ["image", "text_and_image"]:
                self.logger.error(f"Image file not found: {image_data}")
                return None

            # Send the request and get response
            response = self.network.send_and_receive(
                data_type,
                source_lang,
                target_lang,
                image_data=image_data,
                text=text
            )

            if response:
                self.logger.info("Request processed successfully")
            else:
                self.logger.error("Request processing failed")

            return response

    def async_request(self, data_type, source_lang, target_lang, callback=None,
                     image_data=None, text=None):
        """
        Send a request asynchronously.

        Args:
            data_type (str): Type of data ("text", "image", or "text_and_image")
            source_lang (str): Source language code
            target_lang (str): Target language code
            callback (function): Function to call with result
            image_data (str, optional): Path to image file
            text (str, optional): Text content

        Returns:
            Thread: The worker thread handling the request
        """
        def request_task():
            result = self.send_request(data_type, source_lang, target_lang,
                                     image_data, text)
            if callback:
                callback(result)
            return result

        # Create and start the worker thread
        thread = threading.Thread(target=request_task)
        thread.daemon = True

        with self.net_lock:
            self.active_threads.append(thread)
            thread.start()

        return thread

    def get_server_status(self):
        """
        Get the current server connection status.

        Returns:
            dict: Status information including connected state
        """
        with self.net_lock:
            internet_connected = self.update_connection_status()
            server_connected = self.network.connected

            status = {
                "internet_connected": internet_connected,
                "server_connected": server_connected,
                "server_address": f"{self.network.SERVER_IP}:{self.network.SERVER_PORT}" if server_connected else None,
                "last_check": self.last_connection_check
            }

            return status

    def cleanup(self):
        """
        Clean up all network resources before application exit.
        """
        self.logger.info("Cleaning up Network Manager resources")

        # Wait for all active threads to complete
        for thread in self.active_threads:
            if thread.is_alive():
                thread.join(timeout=self.thread_join_timeout)

        self.active_threads.clear()

        # Disconnect from server
        with self.net_lock:
            if self.network:
                self.network.cleanup()
