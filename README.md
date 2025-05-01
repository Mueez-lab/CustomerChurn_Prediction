# 📊 Customer Churn Prediction

This project predicts whether a customer will churn (leave a service) based on various attributes like tenure, internet usage, contract type, and more. It includes complete data preprocessing, EDA, encoding, SMOTE for balancing, and model training with hyperparameter tuning.

## 🔍 Project Overview

- **Objective**: Predict customer churn using machine learning models.
- **Dataset**: Telco Customer Churn Dataset (commonly used for churn analysis)
- **Tech Stack**:
  - Python (Pandas, NumPy, Scikit-learn, Seaborn, Matplotlib)
  - SMOTE (for class imbalance)
  - Models: Decision Tree, Random Forest, XGBoost
  - GridSearchCV for hyperparameter tuning
  - Pickle for model and encoder serialization

## ⚙️ Workflow

1. **Data Cleaning**
   - Dropped unnecessary columns (`customerID`)
   - Replaced empty strings with `NaN`
   - Handled missing values in `TotalCharges`
   - Converted appropriate data types

2. **Exploratory Data Analysis (EDA)**
   - Visualized distributions (histograms, boxplots)
   - Heatmap of correlations
   - Count plots for categorical features

3. **Feature Engineering**
   - Label encoding of categorical variables
   - Stored encoders using `pickle`

4. **Modeling**
   - Applied SMOTE for class balancing
   - Split dataset into training and testing sets
   - Trained three models:
     - Decision Tree
     - Random Forest
     - XGBoost

5. **Hyperparameter Tuning**
   - Used `GridSearchCV` to find the best parameters for each model

6. **Evaluation**
   - Accuracy, Confusion Matrix, and Classification Report
   - Saved final model and encoders for future inference

## 📁 Files Included

- `Customer_Churn_model.pkl`: Trained Random Forest model
- `encoders.pkl`: Label encoders for categorical data
- `customer_churn_prediction.ipynb`: Complete Jupyter Notebook with code
- `README.md`: Project overview and instructions

## 🧪 Sample Inference

```python
input_data = {
    'gender': 'Female',
    'SeniorCitizen': 0,
    'Partner': 'Yes',
    'Dependents': 'No',
    'tenure': 1,
    'PhoneService': 'No',
    'MultipleLines': 'No phone service',
    'InternetService': 'DSL',
    'OnlineSecurity': 'No',
    'OnlineBackup': 'Yes',
    'DeviceProtection': 'No',
    'TechSupport': 'No',
    'StreamingTV': 'No',
    'StreamingMovies': 'No',
    'Contract': 'Month-to-month',
    'PaperlessBilling': 'Yes',
    'PaymentMethod': 'Electronic check',
    'MonthlyCharges': 29.85,
    'TotalCharges': 29.85
}

🚀 Getting Started

    Clone the repository
    Open your terminal or command prompt and run:

git clone https://github.com/Mueez-lab/CustomerChurn_Prediction.git
cd CustomerChurn_Prediction

Install dependencies
(Make sure you have Python installed)

pip install -r requirements.txt

Run the notebook

    Open the Jupyter Notebook:

    jupyter notebook

    Open the customer_churn_prediction.ipynb file.

Make Predictions

    Load the trained model and encoders from the .pkl files provided (Customer_Churn_model.pkl, encoders.pkl).

    Provide your own input data in the dictionary format as shown in the notebook to get predictions.
