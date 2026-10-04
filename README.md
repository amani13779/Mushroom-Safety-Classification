# Mushroom Safety Classification using Machine Learning

**CS435 – Machine Learning**
College of Computer, Department of Computer Science – Qassim University
**Supervisor:** Dr. Renad Alsawed

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Data Preparation](#2-data-preparation)
3. [Exploratory Data Analysis (EDA)](#3-exploratory-data-analysis-eda)
4. [Modeling](#4-modeling)
5. [Optimization](#5-optimization)
6. [Evaluation](#6-evaluation)
7. [Discussion and Conclusion](#7-discussion-and-conclusion)
8. [How to Run](#how-to-run)

---

## 1. Project Overview

This project builds a machine learning model that classifies mushrooms as **edible** or **poisonous** using the Mushroom Dataset. The data was preprocessed, analyzed, and used to train a **Random Forest** classifier. Several evaluation metrics and visualizations were used to measure performance and analyze the predictions.

**Objective:** develop an accurate model that predicts whether a mushroom is edible or poisonous from its features.

**Expected outcomes**

- Build a classification model for mushroom prediction.
- Achieve high accuracy and reliable predictions.
- Identify the important features affecting mushroom classification.
- Evaluate the model using Accuracy, Recall, F1-score, Classification Report, and Confusion Matrix.

---

## 2. Data Preparation

### 2.1 Dataset Description and Source

The project uses the **Mushroom Classification Dataset**, sourced from the UCI Machine Learning Repository and available on Kaggle (license: CC0 Public Domain). It was chosen because it is a clear, practical classification problem with real-world significance.

| Property | Value |
|----------|-------|
| Samples | 8,124 |
| Columns | 23 (22 features + 1 target) |
| Target (`class`) | `e` = edible (4,208), `p` = poisonous (3,916) |
| Problem type | Balanced binary classification |

### 2.2 Data Cleaning and Preprocessing

- All 23 columns are of type `object`, so every feature is **categorical**.
- **No missing values** in any column.
- **No duplicate rows**, so no removal or imputation was needed.
- **Label Encoding** converted all text values to numbers.
- **MinMax normalization** scaled all values to the range 0–1.
- The cleaned dataset was saved as `mushrooms_cleaned.csv`.

---

## 3. Exploratory Data Analysis (EDA)

### 3.1 Statistical Summary and Visualization

All features describe aspects of the mushroom such as cap shape, odor, and gill size. The dataset was explored with `shape`, `info()`, and `describe()`, and bar plots were used to show the distribution of the target and of key variables per class.

- The two classes are fairly balanced.
- Variables like **odor** show clear differences between edible and poisonous mushrooms.

### 3.2 Correlation Analysis and Key Insights

Since the data is categorical, label encoding was applied before computing correlations, and a correlation heatmap was drawn to look at the relationships between the features and the target. Some variables correlate strongly with the class, and **odor** stood out as one of the most important features for deciding whether a mushroom is edible or poisonous.

> Add your charts here: save the figures from the PDF (Odor vs Mushroom Class, Distribution of Mushroom Classes, Correlation Heatmap) into an `images/` folder and embed them like this:
>
> `![Odor vs Class](images/odor_vs_class.png)`

---

## 4. Modeling

The data was split into training and testing sets with an **80/20** ratio (`random_state=42`). A **Random Forest Classifier** was chosen for its strong performance on classification tasks and its ability to handle complex relationships between features. After training, the model was saved with `joblib` as `model.pkl`.

```python
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv("mushrooms.csv")

# Encoding
le = LabelEncoder()
for col in df.columns:
    df[col] = le.fit_transform(df[col])

# Features and target
X = df.drop("class", axis=1)
y = df["class"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Build, train, and save the model
model = RandomForestClassifier()
model.fit(X_train, y_train)
joblib.dump(model, "model.pkl")
```

---

## 5. Optimization

Different hyperparameter values were tested, and the Random Forest models were compared by accuracy. The model with the highest accuracy was selected as the final model.

```python
model1 = RandomForestClassifier(n_estimators=50, max_depth=5)
model2 = RandomForestClassifier(n_estimators=100, max_depth=10)

model1.fit(X_train, y_train)
model2.fit(X_train, y_train)

print("Model 1 Accuracy:", accuracy_score(y_test, model1.predict(X_test)))
print("Model 2 Accuracy:", accuracy_score(y_test, model2.predict(X_test)))

best_model = model2
```

| Model | `n_estimators` | `max_depth` | Accuracy |
|-------|----------------|-------------|----------|
| Model 1 | 50 | 5 | 0.9914 |
| Model 2 (selected) | 100 | 10 | 1.0 |

---

## 6. Evaluation

### 6.1 Metrics

The model was evaluated with scikit-learn metrics:

- **Accuracy:** percentage of correct predictions out of all predictions.
- **Recall:** the model's ability to correctly identify positive cases.
- **F1-score:** combines precision and recall for a balanced evaluation.
- **Classification Report:** precision, recall, F1-score, and support for each class.
- **Confusion Matrix:** correct vs. incorrect predictions, visualized with Seaborn and Matplotlib.

### 6.2 Results

The model achieved **100% Accuracy, Recall, and F1-score** on the test set (1,625 samples).

```text
Accuracy: 1.0
Recall: 1.0
F1-score: 1.0

Classification Report:
              precision    recall  f1-score   support

           0       1.00      1.00      1.00       843
           1       1.00      1.00      1.00       782

    accuracy                           1.00      1625
   macro avg       1.00      1.00      1.00      1625
weighted avg       1.00      1.00      1.00      1625
```

**Confusion matrix**

|  | Predicted 0 | Predicted 1 |
|--|-------------|-------------|
| **Actual 0** | 843 | 0 |
| **Actual 1** | 0 | 782 |

All samples were classified correctly, with no false positives and no false negatives.

---

## 7. Discussion and Conclusion

A Random Forest model was applied to classify mushrooms as edible or poisonous. Preprocessing included removing duplicate rows, Label Encoding of the categorical features, and normalization with `MinMaxScaler`. After testing different hyperparameter settings, the best-performing Random Forest model was selected by accuracy.

The final evaluation showed excellent performance (100% Accuracy, Recall, and F1-score) and a confusion matrix with no wrong predictions. EDA helped identify important patterns, especially the strong relationship between **odor** and the mushroom class.

Overall, the project highlights the importance of data preprocessing, visualization, and model optimization in building accurate machine learning models.

---

## How to Run

1. Download the dataset (`mushrooms.csv`) from the [Mushroom Classification dataset on Kaggle](https://www.kaggle.com/datasets/uciml/mushroom-classification) and place it next to the script.
2. Install the requirements:

```bash
pip install pandas scikit-learn joblib matplotlib seaborn
```

3. Run:

```bash
python mushroom_classification.py
```

The script cleans the data, trains and compares the models, saves `model.pkl`, and prints the evaluation results with the confusion matrix.
