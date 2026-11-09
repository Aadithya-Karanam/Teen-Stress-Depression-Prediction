# Teen Stress & Depression Prediction

## Overview
An ML-based application that analyzes lifestyle, academic, and social media usage patterns to predict depression labels among teenagers.

The project uses a Random Forest Classifier with GridSearchCV for hyperparameter optimization and a Streamlit frontend for user interaction.

## Features
- Predict depression labels based on user inputs.
- Interactive and user-friendly Streamlit interface.
- Data preprocessing and feature encoding.
- Random Forest classification.
- Hyperparameter tuning using GridSearchCV.
- Model evaluation using accuracy, precision, recall, F1-score, and confusion matrix.

## Technologies Used

**Frontend**
- Streamlit

**Backend & Machine Learning**
- Python
- Pandas
- NumPy
- Scikit-learn

**Data Visualization**
- Matplotlib
- Seaborn

## Dataset
The project uses `Teen_Stress.csv`, containing features such as:
- Age and gender
- Daily social media usage
- Platform usage
- Sleep hours
- Screen time before sleep
- Academic performance
- Physical activity
- Social interaction level
- Stress level
- Anxiety level
- Addiction level

**Target Variable:** `depression_label`

## Machine Learning Model

**Algorithm:** Random Forest Classifier

GridSearchCV with 5-fold cross-validation is used for hyperparameter optimization.

Parameters:
- n_estimators: [50, 100, 200]
- max_depth: [None, 5, 10]
- min_samples_split: [2, 5, 10]
- min_samples_leaf: [1, 2, 4]

## Frontend – Streamlit

The application provides an interactive interface where users can enter lifestyle-related information.

### Frontend Features
- Input fields for age and gender.
- Selection of social media platforms.
- Inputs for sleep hours and screen time.
- Academic performance and physical activity inputs.
- Stress, anxiety, and addiction level inputs.
- Prediction button to generate the model output.
- Display of prediction results.

## Project Structure

```text
Teen-Stress-Prediction/
│
├── app.py
├── teen_stress_prediction.py
├── Teen_Stress.csv
├── requirements.txt
├── README.md
└── .gitignore
```

## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/Teen-Stress-Prediction.git
```

Navigate into the project:

```bash
cd Teen-Stress-Prediction
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run the Application

Start the Streamlit frontend:

```bash
streamlit run app.py
```

The application will open in your browser.

## Model Evaluation
The model is evaluated using:
- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix
- Cross-validation score

Add your actual model results here after training.

## Disclaimer
This project is developed for educational and research purposes only. Predictions are based on dataset patterns and should not be considered medical advice or a clinical diagnosis.

## Author
**Karanam Aadithya**

Machine Learning Project