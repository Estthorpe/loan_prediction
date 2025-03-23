## Credit Risk Assessment Dashboard – Machine Learning Project
A dynamic, end-to-end ML solution to predict loan default risk with real-time, tiered risk classification. Built with XGBoost, Streamlit, and SMOTE-enhanced learning.

🚀 **Project Overview**
This project implements a production-ready machine learning pipeline to assess the credit risk of loan applicants. The model outputs a four-tier risk classification — Low, Medium, High, or Extremely High — based on a range of financial, behavioral, and employment-related features.

🎯 **Business Use Case**
Loan defaults cost financial institutions billions yearly. Identifying high-risk borrowers early is critical to reduce losses and improve portfolio quality. This tool:

Predicts the probability of default

Dynamically classifies borrowers into actionable risk segments

Enables loan officers and credit analysts to make informed decisions

🧠 **Key Features**
1.  Dynamic Engineered Features
   -LoanToIncome = LoanAmount / (Income + 1)
   -EmploymentDurationRatio = MonthsEmployed / (Age × 12 + 1)
2. Balanced Training with SMOTE
3. XGBoost Classifier with threshold tuning
4. 4-Level Risk Classification:
    -Low Risk (<30%)
    -Medium Risk (30–50%)
   -High Risk (50–75%)
   -Extremely High Risk (>75%)
5. Streamlit Front-End for live borrower persona scoring
6. Modular scripts for training, testing, and UI integration

**🧪 Model Evaluation**
Accuracy: ~83%
ROC AUC: ~0.68
Recall (Class 1 - Defaulters): ~23%
Confusion Matrix and ROC Curve saved to models/




**📊 Example Output
Borrower Profile	                Default Probability	        Risk Level**
Age 45, Income $150k, PhD	        15.2%	                      🟢 Low Risk
Age 26, Loan $30k, Unemployed	    83.5%	                      🔴 Extremely High Risk


**🧰 Tools & Tech**
. Python, Pandas, NumPy

. XGBoost, SMOTE (imbalanced-learn)

. Scikit-learn, Matplotlib, Seaborn

. Streamlit (for front-end UI)

. Joblib (model serialization)

📤 Future Enhancements
Add SHAP explainability and insights

Deploy to Streamlit Cloud or Hugging Face Spaces

Integrate with SQL database for historical logging

Export PDF/CSV risk reports per applicant





