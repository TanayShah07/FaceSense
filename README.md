# FaceSense 👁️‍🗨️

### Deep Learning Based Biometric Face Recognition System

---

## 📌 Overview

FaceSense is a real-time biometric system that uses **Deep Learning (FaceNet via DeepFace)** to recognize faces and automatically mark attendance.
The system captures facial data, identifies individuals, and logs attendance with timestamps.

---

## 🚀 Features

* Real-time face detection using OpenCV
* Deep learning-based face recognition
* Multi-user support (Tanay, Sonia, etc.)
* Unknown face detection
* Automatic attendance logging (CSV file)
* No duplicate entries in a session

---

## 🧠 Technologies Used

* Python
* OpenCV
* DeepFace (FaceNet model)
* TensorFlow
* NumPy & Pandas

---

## 📂 Project Structure

FaceSense/
│
├── dataset/
│   ├── tanay/
│   ├── sonia/
│
├── outputs/
│   └── attendance.csv
│
├── src/
│   ├── capture_dataset.py
│   ├── face_detection.py
│   └── recognize_face.py
│
├── requirements.txt
└── .gitignore

---

## ⚙️ How to Run

### 1. Activate Virtual Environment

venv\Scripts\activate

### 2. Capture Dataset

python src/capture_dataset.py

Enter person name and capture ~100 images.

### 3. Run Recognition System

python src/recognize_face.py

---

## 📊 Output

* Real-time face recognition displayed on screen
* Attendance stored in:
  outputs/attendance.csv

Example:
Name,Time
tanay,10:32:11
sonia,10:33:05

---

## 🔄 System Workflow

Camera
↓
Face Detection (OpenCV)
↓
Face Embedding (Deep Learning - FaceNet)
↓
Similarity Matching
↓
Recognized / Unknown
↓
Attendance Logging

---

## 🎯 Applications

* Smart attendance systems
* Secure authentication systems
* Office / college monitoring
* Biometric verification

---

## 🧠 Future Improvements

* Mobile app integration
* Cloud-based database
* Faster recognition optimization
* GUI-based interface

---

## 👨‍💻 Authors

Tanay Shah
Sonia

---

## 📌 Note

This project is developed for academic purposes to demonstrate the application of biometrics and deep learning in real-time systems.
