MSSE Machine Learning – Malware Detection CSV Prediction Service
Overview

This project is a machine learning web application developed as part of the Master of Software Systems Engineering (MSSE) – Machine Learning unit.
The system demonstrates the complete lifecycle of a supervised learning model, from training and serialization to deployment as a production-ready web service.

The application allows users to upload a CSV file containing feature data, runs predictions using a trained scikit-learn Pipeline, and returns the results as a downloadable CSV file via a web interface. The service is deployed to Render and includes continuous integration (CI) using GitHub Actions.

Objectives

The primary objectives of this project are to:

Demonstrate correct application of supervised machine learning concepts

Package and serialize a trained scikit-learn model

Deploy a machine learning model as a web service

Provide a usable graphical interface for batch predictions via CSV upload

Implement CI to validate repository integrity

Deploy the service to a cloud hosting environment

Live Deployment

Production URL:
https://msse-machine-learning-otos.onrender.com/predict-csv

The deployed service allows CSV upload, prediction execution, and automatic download of the resulting prediction file.

Features

Web-based CSV upload interface

Batch prediction using a trained scikit-learn Pipeline

Automatic generation of prediction output CSV

Direct file download to the user’s computer

Health check endpoint for service verification

GitHub Actions CI workflow

Cloud deployment using Render

System Architecture

The system follows a simple but effective architecture:

A Flask web application exposes HTTP endpoints

A serialized scikit-learn Pipeline (model.pkl) is loaded at startup

User uploads a CSV file through the web UI

The server processes the CSV and runs model predictions

Predictions are appended to the dataset

The resulting CSV is returned directly to the browser for download

In production, the application uses an ephemeral filesystem, meaning output files are not permanently stored on the server.

Project Structure

MSSE-Machine-Learning

app.py – Flask application and prediction logic

model.pkl – Trained scikit-learn Pipeline

requirements.txt – Python dependencies

templates/index.html – Web UI for CSV upload

downloads/ – Local development output directory

.github/workflows/ci.yml – GitHub Actions CI configuration

README.md – Project documentation

Running the Application Locally
Prerequisites

Python 3.9 or higher

pip

Setup Instructions

Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\Activate.ps1

Install dependencies
pip install -r requirements.txt

Run the Flask application
python app.py

Open the application in a browser
http://127.0.0.1:5000/predict-csv

Usage Instructions

Navigate to the /predict-csv page

Upload a CSV file containing the required feature columns

Click Run Predictions

The browser will automatically download a CSV file containing the prediction results

Health Check Endpoint

Endpoint:
GET /health

Purpose:
Confirms the application is running and responsive.
Used for monitoring and deployment validation.

Model Details

Model Type: scikit-learn Pipeline

Serialization: joblib

Prediction Interface: predict()

The model is validated at startup to ensure compatibility

The use of a Pipeline ensures that preprocessing and prediction logic are consistently applied in both training and inference.

Continuous Integration

Continuous integration is implemented using GitHub Actions.

Workflow file location: .github/workflows/ci.yml

Triggered on push to the main branch

CI is configured to be non-blocking for deployment

This ensures repository consistency without preventing cloud deployment.

Deployment

Platform: Render

Deployment type: Web Service

Automatic deployment on push to main

Application listens on port 10000 in production

Output files are returned via HTTP response rather than stored

Limitations and Assumptions

Uploaded CSV files must match the feature schema used during training

The production environment does not persist files between requests

The service is designed for batch prediction, not real-time streaming

Conclusion

This project demonstrates the successful deployment of a machine learning model as a production web service, incorporating best practices in model serialization, application architecture, CI, and cloud deployment. The system provides a complete and practical example of applying machine learning concepts within a real-world software engineering context.

Author

Bronwyn Trump
Master of Software Systems Engineering
Machine Learning Project


## Environment
- Python 3.9
- numpy 1.26.4
- scikit-learn 1.6.1


Model pickles require `Pipeline.py` and `pipeline.py` shims.