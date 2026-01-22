# Naive Bayes Classification on Ionosphere Dataset

This project applies **Gaussian Naive Bayes** to classify radar signal returns from the
**Ionosphere dataset** as *good* or *bad*. The dataset is widely used to benchmark
classification algorithms on high-dimensional numerical data.

The complete machine learning pipeline — preprocessing, training, prediction, and evaluation —
is implemented in Python using scikit-learn.

---

## 📊 Dataset Information
- **Source**: UCI Machine Learning Repository (Ionosphere)
- **Number of Samples**: 351
- **Number of Features**: 34
  - 32 continuous numeric features
  - 2 boolean features (converted to 0/1)
- **Target Column**: `column_ai`
- **Target Classes**:
  - `g` → Good radar return
  - `b` → Bad radar return


---

## 🎯 Objective

To build and evaluate a **Naive Bayes classifier** that can accurately distinguish
between good and bad radar signal returns.

---

## ⚙️ Technologies Used

- Python 3.x
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn

---

## 🧪 Workflow Implemented

1. Data loading and inspection
2. Feature–target separation
3. Label encoding of target variable
4. Train–test split (70/30)
5. Model training using **Gaussian Naive Bayes**
6. Prediction on test data
7. Model evaluation
   - Accuracy score
   - Classification report
   - Confusion matrix visualization

---

## 🧠 Why Gaussian Naive Bayes?

- All features are continuous → GaussianNB is appropriate
- Works well with high-dimensional data
- Fast training and inference
- Strong baseline model for comparison

---

## 📈 Model Evaluation

The model is evaluated using:
- Accuracy
- Precision, Recall, F1-score
- Confusion Matrix

> Results may vary slightly due to random train-test split.

---

📌 Sample Code Snippet
---
from sklearn.naive_bayes import GaussianNB

model = GaussianNB()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

---
Processing Steps
---
1. Feature–target separation
2. Conversion of boolean features (`column_a`, `column_b`) to integers
3. Label encoding of target variable (`g`, `b`)
4. Train–test split (70/30)
5. Model training using Gaussian Naive Bayes

---
🚀 Future Improvements

Compare with Logistic Regression, SVM, and Random Forest

Apply cross-validation

Feature selection

Hyperparameter tuning

Deploy using Streamlit or Flask

---
Tags
---

Machine Learning Classification Naive Bayes Ionosphere Dataset
Scikit-learn Python Data Science
