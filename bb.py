import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# ✅ Load dataset
df = pd.read_csv("cleaned_data5.csv")

# ✅ Convert numeric columns
num_cols = ['Temperature_Mean', 'Windspeed', 'TSA']
for col in num_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# ✅ Drop rows with missing numeric data
df.dropna(subset=num_cols, inplace=True)

# ✅ Encode target variable
def classify_bleaching(p):
    if p >= 60:
        return "Severe"
    elif p >= 15:
        return "Moderate"
    else:
        return "Mild"

df['Bleaching_Level'] = df['Percent_Bleaching'].apply(classify_bleaching)

# Encode labels as numbers
label_encoder = LabelEncoder()
df['Bleaching_Level_enc'] = label_encoder.fit_transform(df['Bleaching_Level'])

# ✅ Select only relevant features
features = ['Ocean_Name', 'Exposure', 'Temperature_Mean', 'Windspeed', 'TSA']
X = df[features]
y = df['Bleaching_Level_enc']

# ✅ One-hot encode categorical features
X = pd.get_dummies(X, columns=['Ocean_Name', 'Exposure'], drop_first=False)

# ✅ Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ✅ Train Random Forest
rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(X_train, y_train)

# ✅ Test
y_pred = rf.predict(X_test)

# Decode predictions
pred_class = label_encoder.inverse_transform(y_pred)

# Quick summary
print("Sample predictions:", pred_class[:10])
print("Feature columns used:", X_train.columns.tolist())

# Example new coral site(s)
new_sites = pd.DataFrame([{
   'Temperature_Mean': 300.4,
    'Windspeed': 4.6,
    'TSA': -0.8,
    'Ocean_Name': 'Atlantic',
    'Exposure': 'Exposed'
}, {
    'Temperature_Mean': 300.2,
    'Windspeed': 4.4,
    'TSA': 0.1,
    'Ocean_Name': 'Pacific',
    'Exposure': 'Sheltered'
}])

# One-hot encode categorical features (match training columns)
new_sites_encoded = pd.get_dummies(new_sites, columns=['Ocean_Name', 'Exposure'], drop_first=True)

# Add any missing columns (from training) with zeros
for col in X_train.columns:
    if col not in new_sites_encoded.columns:
        new_sites_encoded[col] = 0

# Ensure the column order matches training set
new_sites_encoded = new_sites_encoded[X_train.columns]

# Predict
pred_labels = rf.predict(new_sites_encoded)
pred_classes = label_encoder.inverse_transform(pred_labels)

print("Predicted Bleaching Levels:", pred_classes)
