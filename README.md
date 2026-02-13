# 📌 Customer Churn Prediction using Random Forest (End-to-End ML Project)
📖 Project Overview

This project is an end-to-end Machine Learning pipeline that predicts whether a customer will churn (leave the service) based on their demographic and service usage details.
The goal is to help businesses proactively identify high-risk customers and take retention actions.

This project uses the Telco Customer Churn Dataset and implements a complete ML workflow including:

Data cleaning & preprocessing

Feature engineering

Handling class imbalance

Model training & evaluation

Hyperparameter tuning

Saving final trained model for deployment

🎯 Problem Statement

Customer churn directly impacts revenue in subscription-based businesses.
The objective of this project is to build a predictive model that classifies customers into:

Churn (1) → Customer likely to leave

No Churn (0) → Customer likely to stay

📂 Dataset

Dataset: Telco Customer Churn Dataset
Features include:

Customer tenure

Contract type

Payment method

Internet service

Monthly charges

Total charges
and more.

📌 Target Variable: Churn (Yes/No)

⚠️ Note: Dataset is not uploaded in this repository.
You can download it from Kaggle / GitHub and place it in:

data/telco_churn.csv

🛠 Tech Stack / Tools Used

Python

Pandas, NumPy

Matplotlib

Scikit-learn

Imbalanced-learn (SMOTE)

Joblib

⚙️ Project Workflow
1️⃣ Data Cleaning

Converted TotalCharges to numeric (handled blank values)

Dropped customerID as it is a unique identifier

Converted target column Churn into binary values (Yes=1, No=0)

2️⃣ Preprocessing & Feature Engineering

Implemented using Pipeline + ColumnTransformer to avoid data leakage:

Numerical Features:

Median imputation

Standard scaling

Categorical Features:

Most frequent imputation

OneHotEncoding

3️⃣ Model Training

The main model used is:

✅ RandomForestClassifier

4️⃣ Handling Class Imbalance (Bonus)

Since churn datasets are usually imbalanced, two approaches were implemented:

class_weight="balanced"

SMOTE oversampling

Both approaches were trained and compared.

5️⃣ Hyperparameter Tuning (Bonus)

Used:

✅ GridSearchCV + StratifiedKFold

Scoring metric used:

ROC-AUC

📊 Model Evaluation Metrics

The project evaluates the model using:

Accuracy

Precision

Recall

F1-score

ROC-AUC

Confusion Matrix

ROC Curve

Precision-Recall Curve

All evaluation plots are saved automatically in:

reports/figures/

📌 Results Visualizations

(Once you run the project, these images will be generated automatically)

Confusion Matrix

ROC Curve

Precision-Recall Curve

Feature Importance (Top 20)

📂 Project Structure

customer-churn-prediction-random-forest/
│
├── data/
│   ├── telco_churn.csv  (download manually)
│   └── DOWNLOAD_INSTRUCTIONS.txt
│
├── models/
│   └── churn_rf_model.pkl   (generated after training)
│
├── reports/
│   └── figures/
│       ├── confusion_matrix.png
│       ├── roc_curve.png
│       └── feature_importance.png
│
├── src/
│   └── train.py
│
├── requirements.txt
└── README.md


▶️ How to Run This Project
Step 1: Clone Repository
git clone https://github.com/YOUR_USERNAME/customer-churn-prediction-random-forest.git
cd customer-churn-prediction-random-forest

Step 2: Install Requirements
pip install -r requirements.txt

Step 3: Add Dataset

Download dataset and save as:

data/telco_churn.csv

Step 4: Train Model
python src/train.py

💾 Model Saving

After training, the best tuned model is saved automatically as:

models/churn_rf_model.pkl


This model can be used later for deployment using Streamlit / Flask / FastAPI.

🚀 Future Improvements

Deploy model using Streamlit

Add model explainability using SHAP

Monitor model drift using feature distribution tracking

👨‍💻 Author

Ayush Jha
(Data Science | Machine Learning | Python Developer)

⭐ If you like this project

Give the repo a ⭐ and feel free to connect with me on LinkedIn.
