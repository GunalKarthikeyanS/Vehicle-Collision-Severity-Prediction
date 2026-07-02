# 🚗 Vehicle Collision Severity Prediction
### Machine Learning Pipeline for NYC Motor Vehicle Crash Analysis

A machine learning classification model to predict the severity of vehicle collisions using real-world NYC crash data. Built end-to-end from data preprocessing and feature engineering through model training and evaluation.

---

## 📊 Dataset
**NYC Motor Vehicle Collisions — Crashes**
- Source: NYC Open Data (Motor_Vehicle_Collisions_Crashes)
- Real-world crash records across New York City
- Features include contributing factors, location, time, vehicle types, and casualty counts

---

## 🎯 Problem Statement

Predicting whether a vehicle collision results in a **serious crash** (fatality or injury) based on contributing factors, time, location, and vehicle information. This is a binary classification problem:
- **1 = Serious Crash** (at least one person injured or killed)
- **0 = Non-Serious Crash**

---

## 🔧 Pipeline

```
Raw NYC Crash Data (CSV)
    ↓
Exploratory Data Analysis (EDA)
    ↓
Feature Engineering
  - Remove columns with 90%+ missing values
  - Construct SERIOUS_CRASH target variable
  - Convert datetime features
  - Encode categorical variables
    ↓
Model Training
  - XGBoost Classifier
  - Random Forest Classifier
    ↓
Model Evaluation
  - Accuracy, Precision, Recall, F1
  - Feature importance analysis
    ↓
Key Risk Factor Identification
```

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| ML Models | XGBoost, Random Forest (Scikit-learn) |
| Feature Engineering | Custom preprocessing pipeline |
| Environment | Jupyter Notebook |

---

## 📁 Project Structure

```
Vehicle-Collision-Severity-Prediction/
├── Serious_Injury_modeling.ipynb    ← Main notebook (EDA + modeling)
├── requirements.txt
└── README.md
```

---

## ⚙️ How to Run

### 1. Clone the repository
```bash
git clone https://github.com/GunalKarthikeyanS/Vehicle-Collision-Severity-Prediction.git
cd Vehicle-Collision-Severity-Prediction
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download the dataset
Download from [NYC Open Data](https://data.cityofnewyork.us/Public-Safety/Motor-Vehicle-Collisions-Crashes/h9gi-nx95) and place as:
```
data/Motor_Vehicle_Collisions_-_Crashes_20250430.csv
```

### 4. Run the notebook
```bash
jupyter notebook Serious_Injury_modeling.ipynb
```

---

## 💡 Key Features

- **Target Engineering** — Constructs a composite SERIOUS_CRASH variable from multiple casualty columns
- **Missing Value Handling** — Automatically removes features with 90%+ missing data
- **Contributing Factor Analysis** — Visualizes which factors most frequently lead to serious crashes
- **End-to-End Pipeline** — From raw CSV to trained classifier in one notebook
