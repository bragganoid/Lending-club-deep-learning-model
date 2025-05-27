

---

# 🤖 Lending Club Loan Default Prediction

**Deep Learning with Keras and TensorFlow – Course-End Project**

![TensorFlow](https://img.shields.io/badge/TensorFlow-Used-orange) ![Keras](https://img.shields.io/badge/Keras-Used-red) ![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)

---

## 📌 Project Overview

This project aims to develop a **deep learning model** to predict whether a loan from **Lending Club** will default. Using historical loan data from **2007 to 2015**, the model helps financial institutions evaluate risk and reduce loan default rates.

Given the **highly imbalanced nature** of the dataset and the variety of numerical and categorical features, the challenge lies in proper **feature engineering, data preprocessing, and model tuning**.

---

## 🎯 Objective

Build and evaluate a deep learning model using **Keras with TensorFlow backend** to predict the likelihood of loan default based on historical customer and loan data.

---

## 🏦 Domain

**Finance** – Credit risk assessment using machine learning

---

## 🧾 Dataset Description

The dataset contains the following key features:

| Feature             | Description                                                     |
| ------------------- | --------------------------------------------------------------- |
| `credit.policy`     | 1 if customer meets underwriting criteria, 0 otherwise          |
| `purpose`           | Reason for the loan (e.g., `credit_card`, `debt_consolidation`) |
| `int.rate`          | Interest rate assigned to borrower                              |
| `installment`       | Monthly installment amount                                      |
| `log.annual.inc`    | Natural log of annual income                                    |
| `dti`               | Debt-to-income ratio                                            |
| `fico`              | FICO credit score                                               |
| `days.with.cr.line` | Duration of credit line (in days)                               |
| `revol.bal`         | Revolving balance (credit card unpaid amount)                   |
| `revol.util`        | Utilization of credit line                                      |
| `inq.last.6mths`    | Inquiries in last 6 months                                      |
| `delinq.2yrs`       | 30+ day delinquencies in past 2 years                           |
| `pub.rec`           | Public derogatory records (e.g., bankruptcies)                  |

---

## 🔄 Workflow Overview

### 1. 🔧 Feature Transformation

* Convert **categorical features** (like `purpose`) to **numerical values** using encoding techniques.

### 2. 📊 Exploratory Data Analysis (EDA)

* Understand data distribution and relationships
* Visualize class imbalance
* Explore variable importance via correlation heatmaps

### 3. 🧠 Feature Engineering

* Drop **highly correlated features** to reduce dimensionality and multicollinearity
* Normalize numerical variables to improve model convergence

### 4. 🤖 Modeling

* Build a **Deep Neural Network (DNN)** using **Keras**
* Use **TensorFlow** as the backend
* Apply:

  * Dropout layers for regularization
  * Activation functions like ReLU and sigmoid
  * Appropriate loss functions for classification (`binary_crossentropy`)
* Evaluate using metrics such as:

  * **Accuracy**
  * **Precision / Recall**
  * **AUC-ROC Curve**

---

## 🧪 Model Schematic

> Here's a simplified architecture of the neural network used:

```
Input Layer (Normalized Features)
        ↓
Dense Layer (ReLU Activation)
        ↓
Dropout Layer
        ↓
Dense Layer (ReLU Activation)
        ↓
Dropout Layer
        ↓
Output Layer (Sigmoid Activation)
```

---

## 🛠 Technologies Used

* **Python 3.8+**
* **TensorFlow 2.x**
* **Keras**
* **Pandas, NumPy**
* **Matplotlib, Seaborn** (for EDA)
* **Scikit-learn** (for preprocessing and metrics)

---

## 📁 Project Structure

```
├── data/
│   └── loan_data.csv
├── notebooks/
│   └── LendingClub_DeepLearning.ipynb
├── models/
│   └── best_model.h5
├── README.md
```

---

# 📈 Key Outcomes

* Built and validated a deep learning model to predict loan defaults.
* Addressed data imbalance and feature redundancy.
* Provided insights into the most impactful factors in loan default risk.

---

## ▶️ How to Run

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/lendingclub-loan-default-prediction.git
   cd lendingclub-loan-default-prediction
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Run the notebook:

   ```bash
   jupyter lab notebooks/LendingClub_DeepLearning.ipynb
   ```

---


