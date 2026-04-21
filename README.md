## README.md

# End-to-End Credit Scoring Pipeline

Full-stack machine learning solution for assessing credit risk. The project transforms raw financial data from Kaggle into a deployable **XGBoost**-powered dashboard that calculates credit scores and classifies loan applications.

The pipeline covers everything from deep Exploratory Data Analysis and feature engineering to a live interactive interface built with **Streamlit**.

---

## 🛠️ The Pipeline

### 1. Data Cleaning & EDA
* **Missing Value Imputation:** Handled nulls in external sources and demographic data.
* **Outlier Detection:** Identified and capped anomalies in income and employment duration.
* **Correlation Analysis:** Filtered redundant features to reduce model noise.

### 2. Feature Engineering
Created "Smart Features" that capture the financial health of the borrower more effectively than raw variables:
* **Payment Rate:** Relationship between annual installments and total income.
* **Income per Child:** Adjusted disposable income based on family size.
* **Credit/Goods Ratio:** Percentage of the item price being financed.
* **Retirement Status:** Categorical encoding for employment stability.

### 3. Modeling & Evaluation
After testing multiple algorithms (Logistic Regression, Random Forest, LightGBM), **XGBoost** was selected for its superior performance.
* **Metric Focus:** Optimized for **ROC-AUC** and **Gini** to ensure clear separation between "Good" and "Bad" borrowers.
* **Calibration:** Probabilities are mapped to a human-readable **0–1000 Credit Score** scale.

---

## 💻 Tech Stack

* **Core:** Python (Pandas, NumPy)
* **ML Framework:** XGBoost, Scikit-learn
* **App/UI:** Streamlit
* **Visualization:** Matplotlib, Seaborn

---

### Prerequisites
* Python 3.9+
* `pip` or `conda`

### Installation
1.  **Clone the repo:**
    ```bash
    git clone https://github.com/aviollaz/end-to-end-credit-scoring.git
    cd end-to-end-credit-scoring
    ```
2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Run the Dashboard:**
    ```bash
    streamlit run src/app.py
    ```

---

## 📊 Dashboard Features

* **Real-time Prediction:** Adjust sliders and input fields to see instant score updates.
* **Risk Categorization:** * 🟢 **700+**: Approved (Low Risk)
    * 🟡 **400–699**: Manual Review (Medium Risk)
    * 🔴 **<400**: Rejected (High Risk)
* **Model Explainability:** A horizontal bar chart visualizes the global feature importance, showing which variables (like External Scores or Payment Rates) are driving the model's decisions.
---
*Developed by André Viollaz - Computer Science Student @ UBA*
