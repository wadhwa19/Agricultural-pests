# 🐛 Smart Agriculture: Understanding and Identifying agricultural pests

Using Deep Learning for Sustainable Farming

## Overview

Pest infestations cause significant crop losses in many agricultural regions. Traditional identification methods can be slow and unreliable, often leading to overuse of broad-spectrum pesticides.
This project uses deep learning to classify agricultural pests from images, supporting early detection and promoting more sustainable pest management practices.

## Dataset

The dataset was sourced from Kaggle and contains images of 12 pest categories:

ants, bees, beetle, caterpillar, earthworms, earwig, grasshopper, moth, slug, snail, wasp, weevil

Images feature natural backgrounds, varying lighting conditions, and multiple poses.
Link to dataset:https://www.kaggle.com/datasets/vencerlanz09/agricultural-pests-image-dataset/code

## Model

Two models were trained and evaluated:

A custom CNN model built using TensorFlow/Keras

An EfficientNetB0 transfer learning model to improve generalization

Both models were developed and tested in a Jupyter notebook environment.

## Results
Validation Metrics

Validation Accuracy: 0.8569

Validation Loss: 0.4683

Classification Report

Generated using scikit-learn:

Overall Accuracy: 0.86

Macro F1-score: 0.85

Weighted F1-score: 0.86

A confusion matrix and class-wise precision/recall/F1 metrics were also calculated.

## Streamlit Web App

A Streamlit interface allows users to upload pest images and receive real-time predictions.

To run the app:
streamlit run main.py
Link to web-app: https://saesha-s-agricultural-pests-project.streamlit.app/

## Tools & Libraries

TensorFlow / Keras

EfficientNetB0

NumPy, Pandas

OpenCV

Matplotlib, Seaborn

Streamlit

Pillow

## Future Work

Improve accuracy for visually similar species

Add Grad-CAM visual explanations

Deploy lightweight model for mobile use

## Citations

Dataset:
[1] Kaggle, “Agricultural Pest Image Dataset,” Kaggle.com.

[2] Plant Disease Detection System Introduction | Image Classification Project, YouTube video, Dec. 30, 2023. Available: https://www.youtube.com/watch?v=Wdw7BZP4XrA
. This video provides an overview of a plant disease detection project using image classification, explaining how deep learning models can distinguish between healthy and diseased plant images.


[3] Plant Disease Detection System – Machine Learning Project Playlist, YouTube playlist. Available: https://www.youtube.com/playlist?list=PLvz5lCwTgdXDNcXEVwwHsb9DwjNXZGsoy
. This playlist includes multiple videos detailing setup, model training, and application of machine learning to plant disease detection projects. 

[4] YouTube video with ID RwQ-5v-kIck, YouTube, [Online]. Available: https://youtu.be/RwQ-5v-kIck
