# Age & Gender Detection

A Deep Learning-based age and gender detection application using OpenCV. This project detects faces in an image or a webcam feed and estimates their age and gender using pre-trained Caffe models.

## How It Works

The application leverages three neural networks in OpenCV:
1. **Face Detection Network** (`res10_300x300_ssd_iter_140000.caffemodel`): Detects human faces.
2. **Gender Detection Network** (`gender_net.caffemodel`): Predicts Male or Female.
3. **Age Detection Network** (`age_net.caffemodel`): Classifies age into one of eight buckets.

## Run Locally

To run the application locally on your machine:

1. Clone the repository and install the requirements:
```bash
pip install -r requirements.txt
```
2. Run the Gradio Deployment interface locally:
```bash
python app.py
```
*(Alternatively, you can run the local `cv2.imshow` version with `python age_gender_webcam.py`)*

## One-Click Deployment 🚀

This repository is built for instant deployment on cloud providers. 

To deploy this project to the web for free as a fully functional Hugging Face Space:
1. Go to [Hugging Face Spaces](https://huggingface.co/spaces) and create a new Space.
2. Select **Gradio** as the Space SDK.
3. In the creation menu, choose **"Import from GitHub"** and paste this repository link (`https://github.com/YashwanthNavari/age_gender_detection`). 
4. The cloud servers will automatically install `requirements.txt`, run `app.py`, download the required models, and give you a public URL!
