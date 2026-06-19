# AI Image Upscaler

A powerful, native desktop application designed to enhance image resolution, sharpness, and texture details using the deep neural network **Real-ESRGAN** combined with a modern, interactive user interface.

---

## 👨‍💻 Project & Developer Details
* **Developer:** Yousef Bamdad
* **Contact:** [Arlekcino007](https://t.me/Arlekcino007)
* **License:** MIT License
* **Target OS:** Windows 10 / 11

---

## ✨ Key Features
* **Smart Upscaling:** Increase image resolution by 2x and 4x without blurring or losing structural fidelity.
* **Modern GUI:** Built with a dark-themed, material-inspired layout utilizing the `CustomTkinter` library.
* **Asynchronous Execution (Multithreading):** The AI inference pipeline runs entirely in separate background threads to prevent the user interface from freezing (`UI-Freeze`).
* **Smart Tiling Engine:** Automatically breaks extremely large high-resolution images into smaller processing blocks (Tiles) to avoid CPU/RAM overflow (`Out of Memory` errors).
* **Advanced Monitoring:** Features a live interactive preview monitor with zoom functionality and real-time processing logs.

---

## 🛠 Architecture & Codebase Layout
The project follows a modular, object-oriented multi-layer architecture:
1. **`main.py`:** The main entry point of the software, responsible for lifecycle management, environment initialization, and launching the core application window.
2. **`ui.py`:** The presentation layer. Manages the dashboard tab, logs terminal window, custom zoom controllers, and ensures seamless thread-safe communication between the UI and backend pipelines.
3. **`core.py`:** The AI computational pipeline layer. Handles neural network weight allocation, intelligent padding, tiling assembly arrays, and advanced post-processing using `Lanczos4` interpolation matrixes and `CLAHE` contrast-enhancing filters.

---

## 🚀 Installation & Usage Guide

### Method 1: Desktop Installer (Recommended)
No Python installation is required. Navigate to the **Releases** section on the right side of this repository, download `AI_Image_Upscaler_Setup.exe`, and install it. The setup wizard will automatically create Desktop and Start Menu shortcuts.

### Method 2: Running from Source Code
1. Clone the repository:
   ```bash
   git clone [https://github.com/YousefBamdad/AI-Image-Upscaler.git](https://github.com/YousefBamdad/AI-Image-Upscaler.git)
   cd AI-Image-Upscaler

2- Create and activate a Virtual Environment:
python -m venv .venv
# On Windows:
.venv\Scripts\activate

3- Install dependencies:
pip install -r requirements.txt

4- Run the application:
python main.py

📦 Core Dependencies

customtkinter
Pillow
torch
torchvision
opencv-python
basicsr
realesrgan
