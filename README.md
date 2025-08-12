Iris Species Prediction App 🌸

This repository contains a simple web application that predicts the species of an Iris flower based on its sepal and petal measurements. The application uses a Gaussian Naive Bayes classifier trained on the famous Iris dataset.

Introduction

This project demonstrates a basic machine learning web application built with Flask. Users can input measurements of an Iris flower, and the application will predict whether it's a setosa, versicolor, or virginica species.

Features

Simple User Interface:

A straightforward web form to input flower measurements.

Machine Learning Model: 

Employs a Gaussian Naive Bayes model for classification.

Real-time Prediction: 

Provides instant predictions based on user input.

Files in this Repository

app.py: 

The main Flask application file. It loads the dataset, trains the Naive Bayes model, and handles web requests for the home page and prediction.


data.csv: 

The dataset containing Iris flower measurements (sepal length, sepal width, petal length, petal width) and their corresponding species (target). 


requirements.txt:

Lists the Python dependencies required to run the application. 

templates/: 

(Implied, but not provided in the prompt - you'll need index.html and result.html in this directory)

index.html: 

The HTML template for the home page where users input features.

result.html: 

The HTML template to display the prediction result

Dataset
The data.csv file contains a small sample of the Iris dataset. It includes four features (sepal length, sepal width, petal length, and petal width) and the corresponding target variable, which is the Iris species. 

Dependencies

The application relies on the following Python libraries, as listed in 

requirements.txt: 

Flask:

A micro web framework for Python.

pandas: 

Used for data manipulation and analysis, especially for reading data.csv.

scikit-learn: 

A machine learning library used for the Gaussian Naive Bayes classifier.

Future Enhancements
Error Handling Improvements: More robust error handling for invalid user inputs.

User Interface Enhancements: Improve the look and feel of the web pages using CSS.

Model Persistence: Save and load the trained model instead of retraining on every application start.

More Features: Allow users to upload their own datasets for prediction.

<img width="1366" height="768" alt="Screenshot (59)" src="https://github.com/user-attachments/assets/6517f1e0-cdd9-4702-a6a6-20ba9345b8d2" />

<img width="1366" height="768" alt="Screenshot (60)" src="https://github.com/user-attachments/assets/36524ed4-8646-4083-84af-2c3574a5eb3b" />
