# Loan Approval Prediction App

## Overview

This project is a machine learning–based web application that predicts whether a loan application will be **approved or rejected** based on applicant financial and personal details. The application is built using **Python, Scikit-learn, and Streamlit** and provides an interactive interface for users to enter loan-related information and receive instant predictions.

---

## Objective

The main objective of this project is to develop a predictive model that assists in loan approval decisions by analyzing applicant data such as income, credit score, loan amount, and asset values. The system aims to demonstrate how machine learning can support financial decision-making.

---

## Dataset Description

The dataset used in this project contains loan applicant information with the following attributes:

* Number of dependents
* Education level
* Self-employment status
* Annual income
* Loan amount
* Loan term
* CIBIL (credit) score
* Residential asset value
* Commercial asset value
* Luxury asset value
* Bank asset value
* Loan status (Approved / Rejected)

The dataset is clean, contains no missing values, and includes both numerical and categorical features.

---

## Project Workflow

1. Data collection and understanding
2. Data preprocessing and encoding of categorical variables
3. Feature scaling
4. Training multiple machine learning models
5. Model evaluation and selection of the best-performing model
6. Saving the trained model and scaler
7. Building an interactive web application using Streamlit
8. Predicting loan approval based on user input

---

## Model Selection

Multiple classification algorithms were trained and compared using evaluation metrics. The best-performing model was selected based on overall performance and saved for future predictions.

---

## Application Features

* User-friendly web interface
* Real-time loan approval prediction
* Uses trained machine learning model
* Accepts applicant financial and personal details
* Instant decision output (Approved / Rejected)

---

## Tools and Technologies

* Python
* Pandas and NumPy
* Scikit-learn
* Joblib
* Streamlit
* VS Code

---

## How the Application Works

The user enters loan applicant details through the web interface. The input data is processed using the same preprocessing steps applied during model training. The trained model then predicts whether the loan should be approved or rejected, and the result is displayed instantly.

---

## Use Cases

* Educational machine learning project
* Demonstration of predictive analytics in finance
* Beginner-friendly ML deployment example
* Academic mini project or final-year project

---

## Future Enhancements

* Improve prediction accuracy with advanced models
* Add data visualization and analytics dashboard
* Deploy the application online
* Add probability score and risk level
* Enhance user interface design

---

## Conclusion

This project demonstrates how machine learning can be applied to automate and support loan approval decisions. By combining data preprocessing, model training, and web deployment, the system provides a simple and effective predictive solution.

---
