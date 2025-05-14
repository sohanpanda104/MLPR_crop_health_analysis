import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)
from imblearn.over_sampling import SMOTE
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

# Save class names for 'category' for later use in confusion matrix
category_class_names = label_encoders['category'].classes_

# Handle missing values
imputer = SimpleImputer(strategy='mean')
df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

# Separate features and target
X = df_imputed.drop(columns=['category'])
y = df_imputed['category']

# Split BEFORE applying SMOTE
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply SMOTE ONLY on training set
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# Train Random Forest
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train_res, y_train_res)

# Predict
y_pred = clf.predict(X_test)

# Evaluate
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nF1 Score (weighted):", f1_score(y_test, y_pred, average='weighted'))

# F1 Score per class
print("\nClassification Report (F1 per class):")
print(classification_report(y_test, y_pred, target_names=category_class_names))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=category_class_names)
disp.plot(cmap=plt.cm.Blues, xticks_rotation=45)
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()
