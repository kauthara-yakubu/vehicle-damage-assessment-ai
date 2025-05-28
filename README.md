# CrashCal 🚗💥  
> Vehicle Damage Assessment Using Computer Vision  

---
## Demo



## 🧭 Table of Contents

- [Project Background](#project-background)
- [Executive Summary](#executive-summary)
- [Objective](#objective)
- [Use Case](#use-case)
- [Solution Overview](#solution-overview)
- [Challenges](#challenges)
- [System Architecture](#system-architecture)
- [Model Architecture and Pipeline](#model-architecture-and-pipeline)
- [Tools and Frameworks Used](#tools-and-frameworks-used)
- [Improving the Model](#improving-the-model)

---

## 🧱 Project Background

Vehicle insurance claims still rely heavily on manual inspection, phone calls, and physical paperwork. This process is time-consuming, error-prone, and often inconvenient for both insurers and vehicle owners.

At the same time, modern smartphones and dashcams have made high-quality car photos more accessible than ever. By combining these images with deep learning, we can automate damage classification, reduce human errors, and accelerate the claims approval process — especially for common, visible damage.

CrashCal was developed to prove how **Convolutional Neural Networks (CNNs)** and **Computer Vision** can serve as a scalable foundation for automated, remote damage assessment in real-world insurance workflows.

---

## 📌 Executive Summary

CrashCal is an AI-powered image classification system designed to assess vehicle damage from a single photo. Built with deep learning, it automatically verifies whether an image shows a damaged car, identifies the car brand, classifies the location and severity of the damage, and estimates repair cost.

By streamlining the claims triage process, CrashCal enhances speed, accuracy, and customer satisfaction — all without requiring a physical inspection.

This project is a portfolio-ready demonstration of applying **deep learning for real-world automation** in the insurance and automobile industry.

---

## 🎯 Objective

To build a deep learning model from scratch that automates car damage detection, location classification, and cost estimation — using only an uploaded image as input.

CrashCal simulates an AI-powered claims triage system that reduces manual work for insurers while providing faster, more accurate assessments for users.

---

## 🚘 Use Case

CrashCal is designed to support:

- 🏢 **Insurance Companies** – Automate damage verification and reduce claims processing time  
- 🔧 **Automobile Workshops** – Pre-diagnose repair work and estimate costs before inspection  
- 👤 **Car Owners** – Instantly evaluate damage severity and get remote repair estimates

---

## ✅ Solution Overview

Users upload a car image through the web interface. The system walks the image through a series of classification checkpoints:

- Validates that the image shows a car  
- Detects the **brand** of the car  
- Confirms if visible damage is present  
- Classifies the **location** of the damage (front, rear, or side)  
- Estimates **severity** (minor, moderate, severe)  
- Predicts a cost estimate based on model training

This flow mimics a real-world claims triage process — only faster, smarter, and more scalable.

---

## ⚠️ Challenges

1. **Image Variability** – Differences in lighting, angle, and resolution affect prediction accuracy.  
2. **Data Scarcity** – High-quality, labeled datasets for damage are hard to obtain and often unbalanced.  
3. **Compute Constraints** – Large image datasets require GPUs and considerable training time.  
4. **Edge Cases** – Rare damage types or unclear images may still require human review.  
5. **Cybersecurity Risks** – Hosting sensitive claim data on cloud systems must meet strict compliance.  
6. **Fraud Prevention** – The system needs logic to flag suspicious or repeated claims.

---

## 🖼️ System Architecture

*📷 (Diagram placeholder — you can insert your system architecture image here)*

---

## 🏗️ Model Architecture and Pipeline

The CrashCal pipeline is structured as follows:

1. **User Uploads Image** – Photo of the vehicle is submitted via the web interface.  
2. **Brand Detection** – Identifies the make/model of the vehicle.  
3. **Gate 1 – Car Verification** – Ensures the image is indeed of a car.  
4. **Gate 2 – Damage Detection** – Detects whether damage is visible.  
5. **Location Classification** – Determines where the damage occurred (front, side, rear).  
6. **Severity Classification** – Labels the damage as minor, moderate, or severe.  
7. **Repair Cost Estimation** – Uses learned patterns to provide a cost estimate.  
8. **Output Report** – Results are returned to the user and optionally shared with insurers or workshops.

---

## 🛠️ Tools and Frameworks Used

### 📦 Data Collection
- Google Images  
- Kaggle Vehicle Damage Datasets  
- Import.io (web scraping)

### 🧠 Model Development
- TensorFlow + Keras – Deep learning frameworks  
- NumPy, Scikit-learn – Data handling and preprocessing

### 🌐 Web Application
- Flask – Python web framework  
- Bootstrap – UI styling and layout

### 💻 Development Tools
- Jupyter Notebooks  
- PyCharm  
- Anaconda (virtual environments)

### 📚 Core Python Libraries
- `numpy`, `pandas`, `matplotlib`, `seaborn`, `sklearn`, `pickle`

---

## 🚀 Improving the Model


1. **Mobile Support** – Optimize model size for mobile deployment.
2. ** Brand Addition** - Add more car brand to make the model more robust.
3. **Cloud Integration** – Use cloud APIs for secure and scalable deployment.  
4. **Policy Guidance** – Recommend relevant coverage options based on damage type. 
5. **Explainability** – Add visual tools (e.g., Grad-CAM) to show which parts of the image the model is using for predictions.

---


