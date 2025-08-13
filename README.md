Image Classification App
This project is a Flask-based web application for image classification. It uses a pre-trained MobileNetV2 model to extract features from an uploaded image and then classifies the image using a Support Vector Classifier (SVC) trained on a custom dataset.

Features
Image Upload: A simple web form to upload an image file.

Transfer Learning: Uses the pre-trained MobileNetV2 model as a feature extractor to get powerful, high-level features from images.

Dimensionality Reduction: Applies Principal Component Analysis (PCA) to reduce the dimensionality of the extracted features, improving the efficiency and performance of the classifier.

Image Classification: Employs a Support Vector Classifier (SVC) to predict the class of the input image.

Web Interface: Displays the classification result and, if available, the probability distribution across all classes.

Project Structure
app.py: The main Flask application file. It handles image uploads, processes the images using the loaded models, makes a prediction, and renders the results.

train.py: A script to train the classification pipeline. It loads images from a specified directory (data/train), extracts features using MobileNetV2, applies PCA, trains an SVC, and saves the trained models (pca.pkl, clf.pkl, le_labels.pkl).

requirements.txt: Lists all the necessary Python packages, including Flask, tensorflow, scikit-learn, and Pillow.

models/: This directory stores the trained models and label encoders.

pca.pkl: The trained PCA model.

clf.pkl: The trained SVC classifier.

le_labels.pkl: The label encoder for mapping class names to numerical indices.

uploads/: A temporary directory for storing uploaded images.

data/train/: The directory where your training images are expected to be, organized into subdirectories by class name.

<img width="743" height="442" alt="Screenshot (77)" src="https://github.com/user-attachments/assets/35b554f3-fccf-4320-b477-58bcd6eedd2a" />
<img width="730" height="449" alt="Screenshot (79)" src="https://github.com/user-attachments/assets/818f9c8a-d956-42f3-b700-a61e2891cf52" />

