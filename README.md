# ❤️ Heart Disease Prediction System

A machine learning-based multiclass Heart Disease Prediction System built using the UCI Heart Disease dataset and deployed as an interactive Streamlit web application.

The system accepts clinical patient information, processes the available features, and predicts one of five target classes ranging from Class 0 to Class 4. It also displays the model's predicted probabilities for all five classes.

> ⚠️ **Disclaimer:** This project is intended for educational and demonstration purposes only. It is not a medical diagnostic system and should not be used as a substitute for professional medical advice.

---

## 📌 Project Overview

Heart disease is a major health concern, and machine learning can be used to identify patterns in clinical data that may assist in predictive analysis.

This project develops a multiclass machine learning system that learns from patient clinical attributes and predicts the corresponding heart disease target class.

The project covers the complete machine learning workflow:

- Dataset exploration
- Data quality analysis
- Missing-value analysis
- Feature analysis
- Categorical feature encoding
- Train-test splitting
- Multiclass classification
- Class imbalance handling
- Model comparison
- Feature importance analysis
- Model serialization
- Interactive Streamlit application
- Cloud deployment

---

## 🎯 Objectives

The main objectives of this project are:

1. Analyze the UCI Heart Disease dataset.
2. Understand the structure and quality of the clinical data.
3. Handle categorical variables while preserving missing values.
4. Build a multiclass heart disease prediction model.
5. Address class imbalance during model training.
6. Compare different machine learning algorithms.
7. Identify potentially important predictive features.
8. Save the trained model for reuse.
9. Develop an interactive web interface using Streamlit.
10. Deploy the application so it can be accessed through a public URL.

---

# 📊 Dataset

The project uses the **UCI Heart Disease dataset**.

The combined dataset contains:

- **920 patient records**
- **16 original columns**
- **5 target classes**

### Original Columns

| Feature | Description |
|---|---|
| `id` | Record identifier |
| `age` | Age of the patient |
| `sex` | Sex of the patient |
| `dataset` | Source dataset/cohort |
| `cp` | Chest pain type |
| `trestbps` | Resting blood pressure |
| `chol` | Serum cholesterol |
| `fbs` | Fasting blood sugar |
| `restecg` | Resting electrocardiographic results |
| `thalch` | Maximum heart rate achieved |
| `exang` | Exercise-induced angina |
| `oldpeak` | ST depression induced by exercise |
| `slope` | Slope of the peak exercise ST segment |
| `ca` | Number of major vessels |
| `thal` | Thalassemia |
| `num` | Target class |

---

# 🎯 Target Variable

The original `num` target contains five classes:

| Class | Target |
|---:|---|
| 0 | No Heart Disease |
| 1 | Heart Disease - Class 1 |
| 2 | Heart Disease - Class 2 |
| 3 | Heart Disease - Class 3 |
| 4 | Heart Disease - Class 4 |

The original target values are retained rather than inventing additional clinical severity labels.

The project does **not** interpret Class 1–4 as "mild", "moderate", "severe", or "critical", because those labels are not directly established by the target variable used in this project.

---

# 🔍 Dataset Distribution

The target distribution in the dataset is:

| Class | Number of Records |
|---:|---:|
| 0 | 411 |
| 1 | 265 |
| 2 | 109 |
| 3 | 107 |
| 4 | 28 |

This demonstrates a significant class imbalance, particularly for Class 4.

Because of this imbalance, accuracy alone is not sufficient for evaluating the model. Macro F1-score and class-level precision/recall are also considered.

---

# 🧹 Data Quality Analysis

The dataset was analyzed for:

- Missing values
- Duplicate records
- Invalid numerical values
- Categorical values
- Class imbalance
- Potentially problematic features

### Missing Values

Several clinical attributes contain missing values, including:

- `fbs`
- `restecg`
- `thalch`
- `exang`
- `oldpeak`
- `slope`
- `ca`
- `thal`

Instead of automatically replacing missing medical information with median/mode values, missing values were preserved where possible.

The final machine learning model used, `HistGradientBoostingClassifier`, supports missing values natively.

---

# 🔎 Data Anomalies Investigated

During exploratory analysis, several unusual values were identified.

Examples include:

- A resting blood pressure value of `0`
- Multiple cholesterol values of `0`
- Negative `oldpeak` values

These values were investigated rather than automatically converting every zero or unusual value into a missing value.

This approach was chosen to avoid making unsupported assumptions about the original medical records.

---

# 🧠 Feature Selection

The original dataset contained:

```text
id
age
sex
dataset
cp
trestbps
chol
fbs
restecg
thalch
exang
oldpeak
slope
ca
thal
num
