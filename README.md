# 🛣️ Road Damage Detection Using YOLOv8 and Streamlit

This is a web-based prototype built with Streamlit that uses a YOLOv8 deep learning model to detect and classify road damages from images or videos. The prototype supports 4 classes: longitudinal cracks, transverse cracks, potholes, and alligator cracks.

## 🚀 Features
- Upload and detect road damage from images or videos
- See annotated output with bounding boxes and confidence scores
- Auto-generate a PDF report of detected damages
- Download results easily
- Runs entirely in your browser

## 🧠 Model
- YOLOv8 small (YOLOv8s) trained on a custom dataset of 4 road damage classes

## 🧰 Tech Stack
- Python, Streamlit
- YOLOv8 (Ultralytics)
- OpenCV, Pillow, pandas
- fpdf (for report generation)

## 📝 How to Run
1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/road-damage-detection.git
   cd road-damage-detection

2. Install the dependencies
    pip install -r requirements.txt

3. Run the Streamlit app
    streamlit run app.py


