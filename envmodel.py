# Step 1: Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Step 2: Load data
df = pd.read_csv("cleaned_data5.csv")

# Step 3: Select features and target
features = [
    'Latitude_Degrees', 'Longitude_Degrees', 'Exposure', 'Turbidity',
    'Cyclone_Frequency', 'Temperature_Mean', 'Windspeed', 'SSTA',
    'SSTA_DHW', 'TSA', 'TSA_DHWMean'
]
target = 'Bleaching_condition'

df_model = df[features + [target]].copy()

# Step 4: Convert numeric columns
numeric_cols = [
    'Latitude_Degrees', 'Longitude_Degrees', 'Turbidity', 'Cyclone_Frequency',
    'Temperature_Mean', 'Windspeed', 'SSTA', 'SSTA_DHW', 'TSA', 'TSA_DHWMean'
]
for col in numeric_cols:
    df_model[col] = pd.to_numeric(df_model[col], errors='coerce')

# Step 5: One-hot encode categorical features
df_model = pd.get_dummies(df_model, columns=['Exposure'], drop_first=True)

# Step 6: Drop missing values
df_model.dropna(inplace=True)

# Step 7: Split into features (X) and target (y)
X = df_model.drop(columns=[target])
y = df_model[target]

# Step 8: Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Step 9: Train Random Forest Classifier
rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(X_train, y_train)

# Step 10: Evaluate
y_pred = rf.predict(X_test)
print("Accuracy:", round(accuracy_score(y_test, y_pred), 3))
print("\n Classification Report:\n", classification_report(y_test, y_pred))
print("\n Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# ✅ Ensure new data has exactly same columns
new_data_dict = {
    'Latitude_Degrees': 20.1,
    'Longitude_Degrees': 150.0,
    'Turbidity': 0.07,
    'Cyclone_Frequency': 50,
    'Temperature_Mean': 298,
    'Windspeed': 50,
    'SSTA': 15,
    'SSTA_DHW': 5,
    'TSA': 1.0,
    'TSA_DHWMean': 0.5,
    # include all one-hot encoded columns
    'Exposure_Sometimes': 1,
    'Exposure_Sheltered': 0
}

# 🔁 Build DataFrame with same order as training
new_data = pd.DataFrame([[new_data_dict[col] if col in new_data_dict else 0 for col in X.columns]], columns=X.columns)

# ✅ Predict
predicted_condition = rf.predict(new_data)[0]
print("🌿 Predicted Bleaching Condition:", predicted_condition)


