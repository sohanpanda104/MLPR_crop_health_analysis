import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay, classification_report
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import matplotlib.pyplot as plt

# Load the updated dataset
df = pd.read_csv("temp_hum_rain_paths_indices_filled.csv")

# Drop irrelevant columns
df = df.drop(columns=[
    'FarmID', 'State', 'District', 'Sub-District',
    'SDate', 'HDate', 'geometry', 'dataset', 'tif_path',
    'SowingMonth', 'HarvestMonth'
])

# Encode categorical columns
categorical_cols = ['category', 'Crop', 'CNext', 'CLast', 'CTransp', 'IrriType', 'IrriSource', 'Season']
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

# Handle missing values
imputer = SimpleImputer(strategy='mean')
df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

# Separate features and target
X = df_imputed.drop(columns=['category'])
y = df_imputed['category']

# Split before SMOTE
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Apply SMOTE only on training data
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# Train XGBoost
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
xgb_model.fit(X_train_res, y_train_res)

# Predict and evaluate
y_pred = xgb_model.predict(X_test)

# Accuracy and F1 Score (weighted)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("F1 Score (weighted):", f1_score(y_test, y_pred, average='weighted'))

# F1 Score for each class
print("\nF1 Score for each class:")
f1_scores = f1_score(y_test, y_pred, average=None)  # Without averaging, gives per class F1 scores
for i, score in enumerate(f1_scores):
    print(f"Class {i}: {score:.4f}")

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.show()

# Classification report (includes precision, recall, F1 for each class)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
