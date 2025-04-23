````markdown
# AR AI Assistant - Multimodal Translation System

**Real-time speech/image translation with hybrid cloud/edge processing**

## 🌟 Features

- **Multimodal Input Support**
  🗣️ Speech-to-Speech Translation • 📸 Image Text Extraction & Translation
- **Hybrid Operation Modes**
  🌐 Cloud-connected AI Processing • 📴 Local Edge Computing (Raspberry Pi Optimized)
- **Multi-language Support**
  English ↔ Arabic ↔ French • Expandable Language Framework
- **Smart Interaction**
  Voice Activation ("Hi David") • Context-Aware Processing • Adaptive UI Overlays

## 🛠️ Technology Stack

| Component        | Technologies Used                                   |
| ---------------- | --------------------------------------------------- |
| Core Framework   | Python 3.9, PyQt5, Socket Programming               |
| AI/ML Engine     | Vosk (ASR), HuggingFace Transformers, Tesseract OCR |
| Vision System    | OpenCV, Picamera2, Libcamera                        |
| Audio Processing | Sounddevice, FLite TTS, PulseAudio                  |
| Deployment       | Raspberry Pi OS, ARM-optimized Models               |

## 🚀 Installation

**System Requirements:**

- Raspberry Pi 4 (Recommended) or Linux PC
- Camera Module (for image features)
- Python 3.9+

```bash
# Clone repository
git clone https://github.com/khaledkhedr18/ar_glasses_ai_assistant.git
cd ar_glasses_ai_assistant

# Install Python dependencies
pip install -r requirements.txt

# Install system components
sudo apt-get install -y \
  flite \
  tesseract-ocr \
  tesseract-ocr-ara \
  tesseract-ocr-fra \
  libatlas-base-dev
```
````

## ⚙️ Configuration

**1. Model Setup:**

```bash
# Directory structure
mkdir -p ~/Desktop/gradproj/

# Download Vosk models to:
# ~/Desktop/gradproj/
# - vosk-model-ar-mgb2-0.4
# - vosk-model-small-en-us-0.15
# - vosk-model-small-fr-0.22
```

**2. Network Settings (functions.py):**

```python
SERVER_IP = '192.168.1.65'  # Your translation server IP
SERVER_PORT = 4040           # Match server configuration
```

## 🖥️ Usage

**Launch Application:**

```bash
python main_gui.py
```

**Workflow:**

1. Wake phrase: "Hi David"
2. Choose input type (voice/image)
3. Select source & target languages
4. Provide input:
   - 🎤 Speak naturally for voice translation
   - 📷 Capture text-containing image
5. Receive translated output via speech+display

**Key Commands:**
| Command | Action |
|------------------|----------------------------|
| "Take" | Capture image |
| "Stop"/"Exit" | End session |
| "Yes"/"No" | Confirm/cancel operations |

## 📂 Project Structure

```
AR-AI-Assistant/
├── functions.py          # Core logic (networking, processing)
├── main_gui.py           # PyQt5 GUI entry point
├── model_loader.py       # Model management system
├── vosk-models/          # Speech recognition models
├── server/               # Translation server (optional)
└── requirements.txt      # Dependency specifications
```

## 📦 Dependencies

**Python Packages:**

```
PyQt5==5.15.7
vosk==0.3.45
transformers==4.28.1
opencv-python==4.7.0.72
pytesseract==0.3.10
sounddevice==0.4.6
```

**System Packages:**

```
flite tesseract-ocr libcamera-dev pulseaudio
```

## ⚠️ Important Notes

- **Raspberry Pi Setup:**
  Enable camera interface via `raspi-config`
- **Performance:**
  Minimum 2GB RAM recommended for offline use
- **Security:**
  All network data base64-encoded • No persistent storage of user data
- **First Run:**
  Allow 2-5 minutes for initial model loading

## 🛣️ Roadmap

- [ ] Expand to 10+ languages
- [ ] Mobile companion app integration
- [ ] Sign language recognition module
- [ ] Advanced diarization for multi-speaker scenarios

## 📜 License

Apache 2.0 License | © 2023 [Khaled Khedr]

---

**Contribution Guidelines:**

1. Fork repository
2. Create feature branch
3. Submit PR with detailed documentation
4. Follow PEP8 coding standards

_For commercial use or custom implementations, contact author._
