# Step 1: Import required libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np
import matplotlib.pyplot as plt

# Step 2: Load dataset
df = pd.read_csv("cleaned_data5.csv")

# Select important features# Step 1: Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Step 2: Load your dataset
df = pd.read_csv("cleaned_data5.csv")

# Step 3: Select features and target
features = [
    'Latitude_Degrees', 'Longitude_Degrees', 'Exposure', 'Turbidity',
    'Cyclone_Frequency', 'Temperature_Mean', 'Windspeed', 'SSTA',
    'SSTA_DHW', 'TSA', 'TSA_DHWMean'
]
target = 'Percent_Bleaching'

df_model = df[features + [target]].copy()

# Step 4: Convert numeric columns
numeric_cols = [
    'Latitude_Degrees', 'Longitude_Degrees', 'Turbidity', 'Cyclone_Frequency',
    'Temperature_Mean', 'Windspeed', 'SSTA', 'SSTA_DHW', 'TSA', 'TSA_DHWMean', 'Percent_Bleaching'
]
for col in numeric_cols:
    df_model[col] = pd.to_numeric(df_model[col], errors='coerce')

# Step 5: Encode Exposure and drop missing
df_model = pd.get_dummies(df_model, columns=['Exposure'], drop_first=True)
df_model.dropna(inplace=True)

# Step 6: Create categorical target for severity
def classify_bleaching(p):
    if p >= 60:
        return "Severe"
    elif p >= 15:
        return "Moderate"
    else:
        return "Mild"

df_model["Bleaching_Level"] = df_model["Percent_Bleaching"].apply(classify_bleaching)

# Step 7: Split into features/target
X = df_model.drop(columns=["Percent_Bleaching", "Bleaching_Level"])
y = df_model["Bleaching_Level"]

# Step 8: Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Step 9: Train Random Forest
rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(X_train, y_train)

# Step 10: Evaluate
y_pred = rf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

features = [
    'Latitude_Degrees', 'Longitude_Degrees', 'Exposure', 'Turbidity',
    'Cyclone_Frequency', 'Temperature_Mean', 'Windspeed', 'SSTA',
    'SSTA_DHW', 'TSA', 'TSA_DHWMean'
]
target = 'Percent_Bleaching'

# Subset data
df_model = df[features + [target]].copy()

# Convert numeric columns that might be strings
numeric_cols = [
    'Latitude_Degrees', 'Longitude_Degrees', 'Turbidity', 'Cyclone_Frequency',
    'Temperature_Mean', 'Windspeed', 'SSTA', 'SSTA_DHW', 'TSA', 'TSA_DHWMean', 'Percent_Bleaching'
]
for col in numeric_cols:
    df_model[col] = pd.to_numeric(df_model[col], errors='coerce')

# Encode categorical variable 'Exposure' (creates dummy columns)
df_model = pd.get_dummies(df_model, columns=['Exposure'], drop_first=True)

# Drop missing values (simplest approach)
df_model.dropna(inplace=True)

# Step 3: Split into features (X) and target (y)
X = df_model.drop(columns=['Percent_Bleaching'])
y = df_model['Percent_Bleaching']

# Train-test split (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
# Step 4: Standardize numeric features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 5: Train model
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Step 6: Predict on test set
y_pred = model.predict(X_test_scaled)

# Step 7: Evaluate performance
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("R² Score:", r2)
print("Mean Absolute Error:", mae)
print("Root Mean Squared Error:", rmse)

# Step 8: Plot actual vs predicted
plt.figure(figsize=(7, 5))
plt.scatter(y_test, y_pred, alpha=0.5, color='teal')
plt.xlabel("Actual Percent Bleaching")
plt.ylabel("Predicted Percent Bleaching")
plt.title("Actual vs Predicted Bleaching")
plt.plot([0, 100], [0, 100], color='red', linestyle='--')  # reference line
plt.show()

# Example: Predict for a new coral site
sample = pd.DataFrame([{
    'Latitude_Degrees': 100,
    'Longitude_Degrees': 150.0,
    'Turbidity': 100,
    'Cyclone_Frequency': 50,
    'Temperature_Mean': 400,
    'Windspeed': 5,
    'SSTA': 1.5,
    'SSTA_DHW': 3,
    'TSA': 2,
    'TSA_DHWMean': 1.0,
    'Exposure_Sheltered': 0,
    'Exposure_Sometimes': 1
}])

prediction = rf.predict(sample)[0]
print("Predicted Bleaching Severity:", prediction)
