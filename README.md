<div align="center">

# 🕵️‍♂️ Age & Gender Detection AI

[![Python](https://img.shields.io/badge/Python-3.8+-blue?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.5+-green?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org/)
[![Gradio](https://img.shields.io/badge/Gradio-Powered-orange?style=for-the-badge&logo=gradio&logoColor=white)](https://gradio.app/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

**A Real-Time Deep Learning Vision Application built with OpenCV & Gradio**

<img src="https://capsule-render.vercel.app/api?type=waving&color=38bdf8&height=120&section=header&text=Age%20&%20Gender%20Detection&fontSize=40&animation=fadeIn&fontAlignY=35" width="100%" alt="Header" />

[Explore the Live App](https://yashwanthnavari.github.io/age_gender_detection/) · [Report Bug](https://github.com/YashwanthNavari/age_gender_detection/issues) · [Request Feature](https://github.com/YashwanthNavari/age_gender_detection/issues)

</div>

<br />

> **Note:** We've upgraded this project to run 100% in your browser using Pyodide (WebAssembly) & Gradio Lite! Zero backend servers needed. Total privacy.

---

## ✨ Live Deployment 🚀

Experience the power of real-time computer vision from anywhere, straight in your browser!

<div align="center">
  <a href="https://yashwanthnavari.github.io/age_gender_detection/">
    <img src="https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExdXQ4NDM4NjM1bmNxZDVkcmF3YnJxYnpneXBqbDNyYXUwbDk5YjdyNyZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/L1R1tvI9svkIWwpVYr/giphy.gif" alt="Launch Live App" width="400" style="border-radius:15px; box-shadow: 0 10px 20px rgba(0,0,0,0.5);" />
  </a>
  <br>
  <h3><a href="https://yashwanthnavari.github.io/age_gender_detection/">👉 Click Here to Launch the Live Web App 👈</a></h3>
</div>

---

## 🧠 How It Works under the Hood

The application leverages three powerful neural networks using OpenCV's DNN module:

1. **👤 Face Detection Network** (`res10_300x300_ssd_iter_140000.caffemodel`):
   Quickly identifies human faces within the frame using a Single Shot Multibox Detector (SSD) architecture.
2. **⚧️ Gender Detection Network** (`gender_net.caffemodel`):
   Focuses on the detected face region and predicts the subject's gender (Male/Female).
3. **🎂 Age Detection Network** (`age_net.caffemodel`):
   Classifies the estimated age into one of eight distinct continuous brackets.

---

## 💻 Run Locally

Want to modify the code or run it fully natively on your hardware? Setting up is simple.

### 1. Clone & Install
```bash
git clone https://github.com/YashwanthNavari/age_gender_detection.git
cd age_gender_detection
pip install -r requirements.txt
```

### 2. Download Pre-trained Models
```bash
python download_models.py
```

### 3. Launch the Application
Start the beautiful local web interface built with Gradio:
```bash
python app.py
```
*Alternatively, run the pure OpenCV version without UI overhead:*
```bash
python age_gender_webcam.py
```

---

## ☁️ Deployment Architecture

This project is fully optimized for **Serverless Native Browser Deployment** via **GitHub Pages**. 
By harnessing the power of `@gradio/lite` alongside Pyodide, the application intelligently compiles Python & OpenCV into high-performance WASM binaries, executing deep learning inference fully locally on your device without the need for expensive Hugging Face/GPU servers.

<div align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=38bdf8&height=80&section=footer" width="100%" alt="Footer" />
</div>
