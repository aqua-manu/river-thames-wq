# -*- coding: utf-8 -*-
"""Classification FYP24



from google.colab import drive
drive.mount('/content/drive')

import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("/content/drive/MyDrive/Colab Notebooks/thamesriver.csv")

data.head()

data.columns

data.info()

data.keys()

data.shape[0]

import pandas as pd

df = pd.DataFrame(data, columns=[ 'Gran alkalinity u eq/L', 'Total phosphorus (ug/L)',
                                  'Dissolved chloride (mg Cl/L)', 'Dissolved nitrate (NO3)',
                                  'Dissolved ammonium (NH4) (mg/l)','Dissolved nitrite (mg NO2/L)'])

# Print total records in df
num_records = df.shape[0]
print("Total records in df:", num_records)

df_clean = df.copy()


duplicate_count = df_clean.duplicated().sum()
print(f"Number of duplicate rows in the DataFrame: {duplicate_count}")

df_clean.isna().sum()

total_mv = df_clean.isna().sum().sum()
print (total_mv)

print(df_clean.info())

import pandas as pd
df_clean['Gran alkalinity u eq/L'] = pd.to_numeric(df_clean['Gran alkalinity u eq/L'], errors='coerce')
df_clean['Total phosphorus (ug/L)'] = pd.to_numeric(df_clean['Total phosphorus (ug/L)'], errors='coerce')
df_clean['Dissolved nitrate (NO3)'] = pd.to_numeric(df_clean['Dissolved nitrate (NO3)'], errors='coerce')
df_clean['Dissolved ammonium (NH4) (mg/l)'] = pd.to_numeric(df_clean['Dissolved ammonium (NH4) (mg/l)'], errors='coerce')
df_clean['Dissolved nitrite (mg NO2/L)'] = pd.to_numeric(df_clean['Dissolved nitrite (mg NO2/L)'], errors='coerce')




df_clean.isna().sum()

df_clean = df_clean.dropna()
df_clean.info()

df_clean.isna().sum()

"""### ***EDA FOR INSIGHTS***

***PRE-PROCESSING COMPLETE***
"""

df_clean['Gran alkalinity (mg/L CaCO3)'] = df_clean['Gran alkalinity u eq/L'] / 20

df_clean.info()

print(df_clean.head())

####BINARY CLASSIFICATION

def assign_pollution_label(row):
    # Phosphorus
    if row['Gran alkalinity (mg/L CaCO3)'] >= 50:  # High alkalinity
        if row['Total phosphorus (ug/L)'] <= 173:
            phosphorus_label = 0  # Low
        else:
            phosphorus_label = 1  # High
    else:  # Low alkalinity
        if row['Total phosphorus (ug/L)'] <= 114:
            phosphorus_label = 0  # Low
        else:
            phosphorus_label = 1  # High

    # Nitrate
    if row['Dissolved nitrate (NO3)'] <= 50:
        nitrate_label = 0  # Low
    else:
        nitrate_label = 1  # High

    # Ammonium
    if row['Dissolved ammonium (NH4) (mg/l)'] <= 2.4:
        ammonium_label = 0  # Low
    else:
        ammonium_label = 1  # High

    # Nitrite
    if row['Dissolved nitrite (mg NO2/L)'] <= 0.03:
        nitrite_label = 0  # Low
    else:
        nitrite_label = 1  # High

    # Chloride
    if row['Dissolved chloride (mg Cl/L)'] <= 600:
        chloride_label = 0  # Low
    else:
        chloride_label = 1  # High

    # Returns the highest p label among all parameters
    return max(phosphorus_label, nitrate_label, ammonium_label, nitrite_label, chloride_label)

df_clean['pollution_label'] = df_clean.apply(assign_pollution_label, axis=1)
print(df_clean['pollution_label'].value_counts())

df_clean.count()

X = df_clean.drop('pollution_label', axis=1)


X.head()

y = df_clean['pollution_label']
y.head()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

import seaborn as sns
import matplotlib.pyplot as plt

# Visualize the distribution of each feature
for column in X.columns:
    plt.figure()
    sns.histplot(X[column], kde=True)
    plt.title(column)
    plt.show()

from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np

#  logarithmic transformation to skewed features in the training set
X_train['Gran alkalinity u eq/L'] = np.log1p(X_train['Gran alkalinity u eq/L'])
X_train['Total phosphorus (ug/L)'] = np.log1p(X_train['Total phosphorus (ug/L)'])
X_train['Dissolved nitrate (NO3)'] = np.log1p(X_train['Dissolved nitrate (NO3)'])
X_train['Dissolved ammonium (NH4) (mg/l)'] = np.log1p(X_train['Dissolved ammonium (NH4) (mg/l)'])
X_train['Dissolved nitrite (mg NO2/L)'] = np.log1p(X_train['Dissolved nitrite (mg NO2/L)'])

# Create scaler objects
scaler = MinMaxScaler()  # or StandardScaler()

# transform the training features
X_train_scaled = scaler.fit_transform(X_train)



# logarithmic transformation to skewed features in the testing set
X_test['Gran alkalinity u eq/L'] = np.log1p(X_test['Gran alkalinity u eq/L'])
X_test['Total phosphorus (ug/L)'] = np.log1p(X_test['Total phosphorus (ug/L)'])
X_test['Dissolved nitrate (NO3)'] = np.log1p(X_test['Dissolved nitrate (NO3)'])
X_test['Dissolved ammonium (NH4) (mg/l)'] = np.log1p(X_test['Dissolved ammonium (NH4) (mg/l)'])
X_test['Dissolved nitrite (mg NO2/L)'] = np.log1p(X_test['Dissolved nitrite (mg NO2/L)'])

# Transform the testing features
X_test_scaled = scaler.transform(X_test)

import seaborn as sns
import matplotlib.pyplot as plt


for column in X.columns:
    plt.figure()
    sns.histplot(X[column], kde=True)
    plt.title(column)
    plt.show()

#Hypertuning Logistic Regression
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


# hypertuning
param_grid = {
    'C': [0.01, 0.1, 1, 10, 100],
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear', 'saga'],
    'max_iter': [1000]
}


model = LogisticRegression()

# scoring metrics
scoring = {
    'accuracy': 'accuracy',
    'precision': 'precision',
    'recall': 'recall',
    'f1': 'f1',
    'auc': 'roc_auc'
}

#  grid search with multiple scoring metrics
grid_search = GridSearchCV(model, param_grid, cv=5, scoring=scoring, refit='accuracy')
grid_search.fit(X_train_scaled, y_train)


print("Best hyperparameters:", grid_search.best_params_)
for metric in scoring.keys():
    score = grid_search.cv_results_[f'mean_test_{metric}'][grid_search.best_index_]
    print(f"Best {metric} score: {score}")

# Evaluate the model with the best hyperparameters on the testing set
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, best_model.predict_proba(X_test_scaled)[:, 1])

print("Testing performance with best Logistic Regression hyperparameters:")
print("LR Accuracy:", accuracy)
print("LR Precision:", precision)
print("LR Recall:", recall)
print("LR F1-score:", f1)
print("LR AUC-ROC:", auc)

"""### **Confusion Matrix for Logistic Regression**"""

# Logistic Regression confusion matrix
cm_lr = confusion_matrix(y_test, y_pred)


plt.figure(figsize=(8, 6))
sns.heatmap(cm_lr, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Predicted Negative', 'Predicted Positive'],
            yticklabels=['Actual Negative', 'Actual Positive'])
plt.title('Confusion Matrix - Logistic Regression')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()

"""### **LOGISTIC REGRESSION PRECISION RECALL CURVE**"""



"""### **RANDOM FOREST MODEL**"""

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}


model = RandomForestClassifier(random_state=42)


scoring = {
    'accuracy': 'accuracy',
    'precision': 'precision',
    'recall': 'recall',
    'f1': 'f1',
    'auc': 'roc_auc'
}


grid_search = GridSearchCV(model, param_grid, cv=5, scoring=scoring, refit='accuracy')
grid_search.fit(X_train_scaled, y_train)


print("Best hyperparameters:", grid_search.best_params_)
for metric in scoring.keys():
    score = grid_search.cv_results_[f'mean_test_{metric}'][grid_search.best_index_]
    print(f"Best {metric} score: {score}")



# Evaluate the Random Forest model with the best hyperparameters on the testing set
best_rf_model = grid_search.best_estimator_
y_pred_rf = best_rf_model.predict(X_test_scaled)

accuracy_rf = accuracy_score(y_test, y_pred_rf)
precision_rf = precision_score(y_test, y_pred_rf)
recall_rf = recall_score(y_test, y_pred_rf)
f1_rf = f1_score(y_test, y_pred_rf)
auc_rf = roc_auc_score(y_test, best_rf_model.predict_proba(X_test_scaled)[:, 1])

print("Testing performance with best Random Forest hyperparameters:")
print("RF Accuracy:", accuracy_rf)
print("RF Precision:", precision_rf)
print("RF Recall:", recall_rf)
print("RF F1-score:", f1_rf)
print("RF AUC-ROC:", auc_rf)

"""### **RANDOM FOREST FEATURE IMPORTANCE**"""

feature_importances = best_rf_model.feature_importances_


sorted_indices = feature_importances.argsort()[::-1]


print("\nFeature Importances:")
for index in sorted_indices:
    print(f"{X_test.columns[index]}: {feature_importances[index]}")

feature_importances = best_rf_model.feature_importances_


sorted_indices = feature_importances.argsort()[::-1]


feature_names = [X_test.columns[index] for index in sorted_indices]
sorted_importances = [feature_importances[index] for index in sorted_indices]

plt.figure(figsize=(10, 6))
plt.barh(feature_names, sorted_importances)
plt.title("Feature Importances")
plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.tight_layout()
plt.show()

"""### **CONFUSION MATRIX FOR RANDOM FOREST**"""

import seaborn as sns
import matplotlib.pyplot as plt

# Random Forest confusion matrix
cm_rf = confusion_matrix(y_test, y_pred_rf)

# Create a heatmap using Seaborn
plt.figure(figsize=(8, 6))
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Predicted Negative', 'Predicted Positive'],
            yticklabels=['Actual Negative', 'Actual Positive'])
plt.title('Confusion Matrix - Random Forest')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()

"""### **SUPPORT VECTOR MACHINE MODEL**"""

from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


param_grid = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf'],
    'gamma': ['scale', 'auto']
}


model = SVC(probability=True, random_state=42)


scoring = {
    'accuracy': 'accuracy',
    'precision': 'precision',
    'recall': 'recall',
    'f1': 'f1',
    'auc': 'roc_auc'
}


grid_search = GridSearchCV(model, param_grid, cv=5, scoring=scoring, refit='accuracy')
grid_search.fit(X_train_scaled, y_train)


print("Best hyperparameters:", grid_search.best_params_)
for metric in scoring.keys():
    score = grid_search.cv_results_[f'mean_test_{metric}'][grid_search.best_index_]
    print(f"Best {metric} score: {score}")


best_svm_model = grid_search.best_estimator_
y_pred_svm = best_svm_model.predict(X_test_scaled)

accuracy_svm = accuracy_score(y_test, y_pred_svm)
precision_svm = precision_score(y_test, y_pred_svm)
recall_svm = recall_score(y_test, y_pred_svm)
f1_svm = f1_score(y_test, y_pred_svm)
auc_svm = roc_auc_score(y_test, best_svm_model.predict_proba(X_test_scaled)[:, 1])

print("Testing performance with best Support Vector Machine (SVM) hyperparameters:")
print("SVM Accuracy:", accuracy_svm)
print("SVM Precision:", precision_svm)
print("SVM Recall:", recall_svm)
print("SVM F1-score:", f1_svm)
print("SVM AUC-ROC:", auc_svm)

"""### **CONFUSION MATRX FOR SVM**"""

cm_svm = confusion_matrix(y_test, y_pred_svm)


plt.figure(figsize=(8, 6))
sns.heatmap(cm_svm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Predicted Negative', 'Predicted Positive'],
            yticklabels=['Actual Negative', 'Actual Positive'])
plt.title('Confusion Matrix - SVM')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()

"""### **GRADIENT BOOSTING MODEL**"""

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


param_grid = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.1, 1],
    'max_depth': [3, 5, 7],
    'subsample': [0.8, 1.0]
}


model_gb = GradientBoostingClassifier(random_state=42)


scoring = {
    'accuracy': 'accuracy',
    'precision': 'precision',
    'recall': 'recall',
    'f1': 'f1',
    'auc': 'roc_auc'
}


grid_search_gb = GridSearchCV(model_gb, param_grid, cv=5, scoring=scoring, refit='accuracy')
grid_search_gb.fit(X_train_scaled, y_train)


print("Best hyperparameters:", grid_search_gb.best_params_)
for metric in scoring.keys():
    score = grid_search_gb.cv_results_[f'mean_test_{metric}'][grid_search_gb.best_index_]
    print(f"Best {metric} score: {score}")


best_gb_model = grid_search_gb.best_estimator_
y_pred_gb = best_gb_model.predict(X_test_scaled)

accuracy_gb = accuracy_score(y_test, y_pred_gb)
precision_gb = precision_score(y_test, y_pred_gb)
recall_gb = recall_score(y_test, y_pred_gb)
f1_gb = f1_score(y_test, y_pred_gb)
auc_gb = roc_auc_score(y_test, best_gb_model.predict_proba(X_test_scaled)[:, 1])

print("Testing performance with best Gradient Boosting hyperparameters:")
print("GB Accuracy:", accuracy_gb)
print("GB Precision:", precision_gb)
print("GB Recall:", recall_gb)
print("GB F1-score:", f1_gb)
print("GB AUC-ROC:", auc_gb)

"""### **CONFUSION MATRIX GRADIENT BOOSTING**"""

from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


cm_gb = confusion_matrix(y_test, y_pred_gb)


plt.figure(figsize=(8, 6))
sns.heatmap(cm_gb, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Predicted Negative', 'Predicted Positive'],
            yticklabels=['Actual Negative', 'Actual Positive'])
plt.title('Confusion Matrix - Gradient Boosting')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()