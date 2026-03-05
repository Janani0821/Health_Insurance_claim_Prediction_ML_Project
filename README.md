
# Health Insurance Claim Predictor 🏥

An end-to-end machine learning project that predicts medical insurance claim amounts based on patient demographics and health metrics. This repository includes the complete workflow from data analysis and model training to a fully functional, interactive web dashboard.

## Features
* **Predictive Modeling:** Utilizes an optimized XGBoost model to accurately forecast insurance costs based on patient profiles.
* **Data Preprocessing Pipeline:** Implements robust feature scaling and categorical label encoding (for Gender, Diabetes, and Smoking Status) using Scikit-learn to clean and prepare raw health data.
* **Interactive Dashboard:** Features a responsive, user-friendly web interface built with Streamlit, allowing users to input patient metrics and receive real-time payment estimates.

## Project Structure
* **`model.ipynb`**: A Jupyter Notebook detailing the exploratory data analysis (EDA), data preprocessing steps, and the evaluation of multiple machine learning algorithms before selecting the best-performing XGBoost model.
* **`Dashboard.py`**: The Streamlit web application script that handles the user interface and integrates the preprocessors and predictive model.
* **`best_model.pkl`**: The exported XGBoost machine learning model.
* **`scaler.pkl` & `*_label_encoder.pkl` files**: Saved Scikit-learn preprocessors that ensure real-time user inputs are transformed to match the original training data structure.

## Tech Stack
* **Language:** Python
* **Data Manipulation & Visualization:** Pandas, NumPy, Matplotlib, Seaborn
* **Machine Learning:** Scikit-learn, XGBoost, Joblib
* **Web Framework:** Streamlit

## How to Run the App

**1. Clone the repository**
```bash
git clone [https://github.com/YourUsername/your-repo-name.git](https://github.com/YourUsername/your-repo-name.git)
cd your-repo-name
