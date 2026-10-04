import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, recall_score, f1_score,
                             classification_report, confusion_matrix)

# ---------------------------------------------------------------
# 1. Load data and clean
# ---------------------------------------------------------------
df = pd.read_csv("mushrooms.csv")
print("Shape:", df.shape)
print("Missing values:\n", df.isnull().sum())
print("Duplicated rows:", df.duplicated().sum())
df = df.drop_duplicates()

# ---------------------------------------------------------------
# 2. Encoding (all columns are categorical)
# ---------------------------------------------------------------
le = LabelEncoder()
for col in df.columns:
    df[col] = le.fit_transform(df[col])

# Features and target
X = df.drop("class", axis=1)
y = df["class"]

# ---------------------------------------------------------------
# 3. Train / test split (80/20)
# ---------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------------------------------------------------------
# 4. Baseline model
# ---------------------------------------------------------------
model = RandomForestClassifier()
model.fit(X_train, y_train)
joblib.dump(model, "model.pkl")

# ---------------------------------------------------------------
# 5. Optimization: compare hyperparameter settings
# ---------------------------------------------------------------
model1 = RandomForestClassifier(n_estimators=50, max_depth=5)
model2 = RandomForestClassifier(n_estimators=100, max_depth=10)

model1.fit(X_train, y_train)
model2.fit(X_train, y_train)

pred1 = model1.predict(X_test)
pred2 = model2.predict(X_test)

print("Model 1 Accuracy:", accuracy_score(y_test, pred1))
print("Model 2 Accuracy:", accuracy_score(y_test, pred2))

best_model = model2

# ---------------------------------------------------------------
# 6. Evaluation
# ---------------------------------------------------------------
y_pred = best_model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1-score:", f1_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
