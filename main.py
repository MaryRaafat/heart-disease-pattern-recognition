import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

df = pd.read_csv("heart_disease_risk_dataset_earlymed.csv")
print(df.head())
print(df.shape)
print(df.shape[0])
    
X = df.drop("Heart_Risk", axis=1)
y = df["Heart_Risk"]

# تقسيم لتدريب واختبار (80% تدريب - 20% اختبار)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Feature Selection
# ==========================================
selector = SelectKBest(score_func=f_classif, k=8)
X_train_fs = selector.fit_transform(X_train, y_train)
X_test_fs = selector.transform(X_test)


selected_cols = X.columns[selector.get_support()]
print("\nSelected 8 Features:", list(selected_cols))

# Scalling

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_fs)
X_test_scaled = scaler.transform(X_test_fs)

# Feature Extraction
pca = PCA(n_components=5)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)


# Models

# Logistic Regression
lr_model = LogisticRegression()
lr_model.fit(X_train_scaled, y_train)
lr_pred = lr_model.predict(X_test_scaled)

# Random Forest 
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_pca, y_train)
rf_pred = rf_model.predict(X_test_pca)

# Evaluation

def display_results(y_true, y_pred, model_name):
    print(f"\n{'='*30}")
    print(f"RESULTS FOR: {model_name}")
    print(f"{'='*30}")
    
    acc = accuracy_score(y_true, y_pred)
    print(f"Accuracy: {acc*100:.2f}%")
    
#(Precision, Recall, F1)
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))
    
   #Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', cbar=False)
    plt.title(f'Confusion Matrix: {model_name}')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()


display_results(y_test, lr_pred, "Logistic Regression (Feature Selection)")
display_results(y_test, rf_pred, "Random Forest (PCA Extraction)")