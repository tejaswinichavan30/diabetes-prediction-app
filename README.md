# 🩺 Diabetes Prediction App

A Flask-based web application that predicts the likelihood of diabetes using a Machine Learning model trained on medical data. The application provides an easy-to-use interface where users can enter health-related information and receive an instant prediction.

---

## Features

- Predicts diabetes risk using a trained Machine Learning model.
- User-friendly web interface built with Flask.
- Real-time prediction based on user input.
- Data preprocessing using a saved scaler.
- Lightweight and easy to run locally.

---

## Technologies Used

- Python
- Flask
- Scikit-learn
- NumPy
- Pandas
- HTML
- CSS
- Pickle

---

## Project Structure

```
diabetes-prediction-app/
│── app.py
│── train_model.py
│── diabetes_model.pkl
│── scaler.pkl
│── columns.pkl
│── templates/
│── static/
│── README.md
```

---

## Installation

### Clone the repository

```bash
git clone https://github.com/tejaswinichavan30/diabetes-prediction-app.git
```

### Navigate to the project folder

```bash
cd diabetes-prediction-app
```

### Install dependencies

```bash
pip install flask flask-login flask-sqlalchemy scikit-learn pandas numpy joblib
```

### Run the application

```bash
python app.py
```

Open your browser and visit:

```
http://127.0.0.1:5000
```

---

## Machine Learning Model

The application uses a trained Scikit-learn classification model to predict whether a patient is likely to have diabetes based on the provided health parameters.

---

## Future Improvements

- User authentication
- Prediction history
- Data visualization dashboard
- Cloud deployment
- Enhanced model performance

---

## Author

**Tejaswini Chavan**

GitHub: https://github.com/tejaswinichavan30
