# Fraud Detection in Financial Transactions (Paysim Dataset)

_A supervised machine learning project for the DAP391m course, focusing on identifying fraudulent transactions from a highly imbalanced dataset and deploying an interactive analysis dashboard using Streamlit._

---

## 📋 Table of Contents

* [1. Problem Statement](#1-problem-statement)
* [2. Dataset](#2-dataset)
* [3. Project Pipeline](#3-project-pipeline)
* [4. Key Findings & Results](#4-key-findings--results)
* [5. Tech Stack](#5--tech-stack)
* [6. How to Run (Local Setup)](#6--how-to-run-local-setup)
* [7. Dashboard Demo](#7--dashboard-demo)

---

## 1. Problem Statement

As e-commerce and digital payments (like those on platforms similar to MoMo, Shopee, etc.) grow, so does the sophistication of financial fraud. Malicious actors use stolen credentials, create fake transactions for money laundering, or exploit promotional systems.

Traditional rule-based systems (e.g., "flag transactions over $10,000") are often ineffective because they:
* Fail to capture new, unknown fraud patterns.
* Generate a high number of false positives, increasing the manual review workload.

**Project Goal:** The objective of this project is to build and evaluate robust machine learning models that can accurately classify transactions as either **Legitimate** or **Fraudulent**. While the problem is a form of anomaly detection, we utilize a **supervised learning approach** (using the `isFraud` label) to train classifiers on this behavior.

## 2. Dataset

* **Source:** [PaySim1 - Kaggle Dataset](https://www.kaggle.com/datasets/ealaxi/paysim1)
* **Description:** This is a synthetic dataset generated using the PaySim simulator, which mimics real-world mobile money transactions.
* **Key Features:**
    * `step`: Maps a unit of time (1 step = 1 hour).
    * `type`: Type of transaction (e.g., `CASH_OUT`, `TRANSFER`, `PAYMENT`).
    * `amount`: The value of the transaction.
    * `oldbalanceOrg`, `newbalanceOrig`: Sender's balance before and after.
    * `oldbalanceDest`, `newbalanceDest`: Receiver's balance before and after.
    * `isFraud`: The target label (1 for fraud, 0 for legitimate).

The most significant challenge of this dataset is its **extreme class imbalance**. Fraudulent transactions represent only a tiny fraction (~0.1%) of the total data, making detection difficult.

## 3. Project Pipeline

The project was executed in the `notebooks/1_Fraud_Detection_EDA_and_Modeling.ipynb` file and follows these main stages:

1.  **Exploratory Data Analysis (EDA):**
    * Analyzed the distribution of transaction types and amounts.
    * Confirmed the severe class imbalance.
    * Discovered key patterns: fraudulent activity **only occurs** in `TRANSFER` and `CASH_OUT` transaction types.

2.  **Feature Engineering:**
    * Created new features to better capture suspicious behavior, including:
        * `errorBalanceOrig`: Discrepancy between `oldbalanceOrg` - `amount` and `newbalanceOrig`.
        * `errorBalanceDest`: Discrepancy between `oldbalanceDest` + `amount` and `newbalanceDest`.
        * `hour_of_day`: Extracted from the `step` feature to check for temporal patterns.
    * One-hot encoded the categorical `type` feature.

3.  **Data Preprocessing:**
    * Addressed the extreme class imbalance using **SMOTE (Synthetic Minority Over-sampling Technique)**. This was applied *only to the training set* to create synthetic fraud examples and balance the data, preventing the model from being biased towards the majority class.
    * Scaled numerical features using `StandardScaler`.

4.  **Modeling (Supervised Learning):**
    * Trained and compared several supervised classifiers, including Logistic Regression, Decision Tree, and Random Forest.
    * The models were trained on the SMOTE-balanced training data.

5.  **Model Evaluation:**
    * Due to the imbalance, **Accuracy** is a poor metric.
    * Primary evaluation metrics were **F1-Score, Precision, Recall,** and the **Confusion Matrix**.
    * The final chosen model (e.g., Random Forest) was selected based on its high F1-Score, indicating a good balance between precision and recall.

6.  **Deployment:**
    * Developed an interactive dashboard using **Streamlit** (`app/app.py`).
    * The dashboard allows for data exploration, model performance review, and live prediction on new data (both single and batch).

## 4. Key Findings & Results

* **EDA Finding:** 100% of fraudulent transactions occurred in `TRANSFER` and `CASH_OUT` types. This allowed us to significantly filter the data and focus the model on relevant transactions.
* **Feature Importance:** Features engineered to capture balance errors (`errorBalanceOrig`, `errorBalanceDest`) were among the most predictive indicators of fraud.
* **Model Performance:** The final model (e.g., Random Forest) trained on the SMOTE-balanced data achieved an **F1-Score of 0.94 using Linear Regression** on the unseen test set, demonstrating its effectiveness in identifying fraudulent cases without overwhelming operators with false positives.
* **Challenge:** The core challenge was the 1:1000 imbalance ratio. Using SMOTE was critical to achieving high recall.

## 5. 🛠️ Tech Stack

* **Data Analysis & ML:** Python, Pandas, NumPy, Scikit-learn, Imbalanced-learn (for SMOTE)
* **Data Visualization:** Matplotlib, Seaborn, Plotly
* **Web Dashboard:** Streamlit
* **Development Environment:** Jupyter Notebook, VS Code

## 6. 🚀 How to Run (Local Setup)

Follow these steps to run the project on your local machine.

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/financial-fraud-dashboard.git](https://github.com/your-username/financial-fraud-dashboard.git)
    cd financial-fraud-dashboard
    ```

2.  **Create a virtual environment and activate it:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install the required libraries:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Download the Dataset (CRITICAL STEP):**
    * Download the dataset from [Kaggle](https://www.kaggle.com/datasets/ealaxi/paysim1).
    * Unzip and place the `PS_2017...csv` file inside the `/data/` directory. (This directory is ignored by `.gitignore`).

5.  **Run the Streamlit Dashboard:**
    ```bash
    streamlit run app/app.py
    ```
    Open `http://localhost:8501` in your browser.

6.  **(Optional) Run the Jupyter Notebook:**
    If you want to re-run the analysis or model training:
    ```bash
    jupyter notebook notebooks/Fraud_Detection.ipynb
    ```

## 7. 📸 Dashboard Demo

The Streamlit dashboard (`app/app.py`) showcases the project's analysis and model results. The currently functional tabs include:

### Tab 1: 📁 Data & Analysis
* Displays the raw dataset.
* Shows interactive charts (e.g., distribution of `step` or `type`).

`![Data Analysis Tab](images/Tab1.png)`

### Tab 2: 📈 Model Evaluation
* Displays the performance of the saved model on the test set.
* Shows the **Confusion Matrix** and **Classification Report** (Precision, Recall, F1-Score).

`![Model Evaluation Tab](images/Tab2.png)`

### Tab 3: 🤖 Prediction (Work in Progress)
* *This tab is currently under development.*
* **Planned Functionality:** To allow real-time single transaction prediction and batch prediction from an uploaded CSV file.