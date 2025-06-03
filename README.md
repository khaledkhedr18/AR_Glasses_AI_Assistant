# Project Functionalities and LLM Interaction Flow

This document describes the project's core functionalities and how the `LLM_Manager` abstracts and encapsulates interactions with Large Language Models (LLMs) through specialized handlers.

## 1. Project Overview

This project provides a structured way to interact with various Large Language Models for different tasks. It uses a central manager (`LLM_Manager`) to create and provide access to specialized "handler" modules, each responsible for a specific functionality like audio transcription or other model-specific operations.

## 2. Core Components

### 2.1. Configuration
*   **File:** [`configrations/config.py`](configrations/config.py)
*   **Purpose:** Manages all application-level configurations, such as API keys, LLM API endpoints (e.g., `API_TRANSCRIBE_ENDPOINT`), file paths (e.g., `AUDIO_PATH`), and other settings.

### 2.2. LLM Manager (`LLM_Manager.py`)
*   **File:** [`Managers/LLM_Manager.py`](Managers/LLM_Manager.py)
*   **Role:** Acts as a **Registry and Factory** for LLM handlers.
*   **Key Functionalities:**
    *   **Handler Registration:**
        *   The [`LLM_Manager.register_handler(purpose_key, handler_class, is_default=False)`](Managers/LLM_Manager.py:17) class method allows different handler classes (which must be subclasses of `LLM_Manager`) to be registered with a unique string identifier called `purpose_key`.
        *   This creates a central registry mapping `purpose_key`s to their respective handler classes.
        *   A default handler can also be specified.
    *   **Handler Instantiation (Abstraction):**
        *   The [`LLM_Manager.get_instance(purpose_key=None, *args, **kwargs)`](Managers/LLM_Manager.py:28) class method is used to obtain an instance of a specific handler.
        *   Client code requests a handler by its `purpose_key` without needing to know the concrete class name of the handler. If no `purpose_key` is provided, the default handler is instantiated.
        *   The `LLM_Manager` looks up the `purpose_key` and returns a new instance of the associated handler class, passing the `purpose_key` itself as a `purpose` argument to the handler's constructor.

### 2.3. LLM Request Handlers
*   **Directory:** [`Server_LLMS_Requests_Handlers/`](Server_LLMS_Requests_Handlers/)
*   **Role:** Contains specialized modules, each inheriting from `LLM_Manager`, that encapsulate the logic for a specific LLM-related task.
*   **Examples:**
    *   **Audio Transcription Handler:**
        *   **File:** [`Server_LLMS_Requests_Handlers/AudioTranscriptionRequestHandler.py`](Server_LLMS_Requests_Handlers/AudioTranscriptionRequestHandler.py)
        *   **Inherits from:** `LLM_Manager`.
        *   **Functionality Encapsulated:** Handles audio transcription. The [`send_audio_to_api(audio_path)`](Server_LLMS_Requests_Handlers/AudioTranscriptionRequestHandler.py:11) method contains the logic to take an audio file path, send it to a transcription API (defined in `config.py`), and process the response.
    *   **GPM Request Handler:**
        *   **File:** [`Server_LLMS_Requests_Handlers/GpmRequestHandler.py`](Server_LLMS_Requests_Handlers/GpmRequestHandler.py)
        *   **Inherits from:** `LLM_Manager` (presumably, based on the pattern).
        *   **Functionality Encapsulated:** Manages requests for a "GPM" model. It would contain methods specific to interacting with this model.

### 2.4. Utility/Testing
*   **File:** [`try.py`](try.py) - Likely for testing and experimentation.
*   **File:** [`recording.wav`](recording.wav) - Sample audio for testing transcription.

## 3. Abstraction and Encapsulation Flow

The `LLM_Manager` plays a crucial role in abstracting the creation of handlers and allowing handlers to encapsulate their specific tasks:

1.  **Initialization/Setup (e.g., in `try.py` or an application entry point):**
    *   Different handler classes (e.g., `AudioTranscriptionRequestHandler`, `GpmRequestHandler`) are registered with the `LLM_Manager` using their unique `purpose_key`.
    ```python
    # Example of registration (likely done once at startup)
    # from Managers.LLM_Manager import LLM_Manager
    # from Server_LLMS_Requests_Handlers.AudioTranscriptionRequestHandler import AudioTranscriptionRequestHandler
    # from Server_LLMS_Requests_Handlers.GpmRequestHandler import GpmRequestHandler
    #
    # LLM_Manager.register_handler("audio_transcribe", AudioTranscriptionRequestHandler, is_default=True)
    # LLM_Manager.register_handler("gpm_process", GpmRequestHandler)
    ```

2.  **Client Request for Functionality:**
    *   When a part of the application needs a specific LLM functionality, it requests a handler instance from `LLM_Manager` using the `purpose_key`.
    ```python
    # Example: Getting an audio transcription handler
    # audio_handler = LLM_Manager.get_instance("audio_transcribe", audio_path="path/to/specific_audio.wav")
    #
    # Example: Getting the default handler if "audio_transcribe" was default
    # default_audio_handler = LLM_Manager.get_instance(audio_path="path/to/default_audio.wav")
    ```
    *   The `LLM_Manager` instantiates and returns the correct handler object (e.g., an `AudioTranscriptionRequestHandler` instance). The client code is decoupled from the concrete `AudioTranscriptionRequestHandler` class.

3.  **Handler Executes Task:**
    *   The client code then calls methods on the obtained handler instance to perform the task.
    ```python
    # if audio_handler:
    #     transcription_result = audio_handler.send_audio_to_api()
    #     # Process result
    ```
    *   The handler (e.g., `AudioTranscriptionRequestHandler`) uses its encapsulated logic and configurations (from `config.py`) to interact with the actual LLM service and return the result.



## 4. How to Use (Conceptual)

1.  **Configure:** Ensure [`configrations/config.py`](configrations/config.py) has the correct API endpoints, keys, and paths.
2.  **Register Handlers:** In your application's entry point or setup phase, register all available handler classes with the `LLM_Manager` using appropriate `purpose_key`s.
3.  **Get Handler Instance:** When needed, use `LLM_Manager.get_instance("your_purpose_key", ...any_handler_specific_args...)` to get a handler.