# Step 1: Install required packages
!pip install -q openpyxl imbalanced-learn

# Step 2: Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from imblearn.over_sampling import SMOTE
from google.colab import files

# Step 3: Upload Excel file
print("Upload your Excel dataset file:")
uploaded = files.upload()

# Step 4: Load the Excel file
df = pd.read_excel(next(iter(uploaded)))

# Step 5: Display column names to confirm structure
print("Detected columns:", df.columns.tolist())

# Step 6: Automatically detect the target column
target_col = None
for col in df.columns:
    if col.strip().lower() in ['class', 'target', 'label']:
        target_col = col
        break

if not target_col:
    raise ValueError("Target column not found. Please rename your label column to 'Class' or 'Target'.")

# Step 7: Split features and target
X = df.drop(target_col, axis=1)
y = df[target_col]

# Step 8: Scale features
X_scaled = StandardScaler().fit_transform(X)

# Step 9: Handle class imbalance
X_res, y_res = SMOTE(random_state=42).fit_resample(X_scaled, y)

# Step 10: Split data
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, stratify=y_res, random_state=42)

# Step 11: Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 12: Evaluate the model
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("ROC AUC Score:", roc_auc_score(y_test, y_proba))
