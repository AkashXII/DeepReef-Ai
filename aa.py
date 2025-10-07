import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib

class CoralHealthPredictor:
    def _init_(self):
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
        # ============ EDIT THIS: Define your input features ============
        # These should match the column names in your CSV file
        self.features = [
            'Latitude_Degrees', 'Longitude_Degrees', 'Exposure', 'Turbidity',
    'Cyclone_Frequency', 'Temperature_Mean', 'Windspeed', 'SSTA',
    'SSTA_DHW', 'TSA', 'TSA_DHWMean'
        ]
        # ================================================================
        
        self.target = 'Percent_Bleaching'  # This should be your target column name
        
    def load_data(self, csv_path):
        """
        Load training data from CSV file
        
        Args:
            csv_path (str): Path to your CSV file
            
        Returns:
            DataFrame: Loaded data
        """
        print(f"Loading data from {csv_path}...")
        df = pd.read_csv(csv_path)
        print(f"Data loaded successfully! Shape: {df.shape}")
        print(f"\nColumns found: {list(df.columns)}")
        print(f"\nFirst few rows:\n{df.head()}")
        return df
    
    def preprocess_data(self, df):
        """
        Preprocess the data: handle missing values, encode labels, scale features
        
        Args:
            df (DataFrame): Raw data
            
        Returns:
            tuple: X_train, X_test, y_train, y_test
        """
        print("\nPreprocessing data...")
        
        # Check for missing values
        if df.isnull().sum().any():
            print("Warning: Missing values detected. Filling with median values...")
            df = df.fillna(df.median(numeric_only=True))
        
        # Separate features and target
        X = df[self.features]
        y = df[self.target]
        
        # Encode severity labels (Low, Moderate, High) to numbers
        y_encoded = self.label_encoder.fit_transform(y)
        print(f"Severity classes: {self.label_encoder.classes_}")
        
        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        print(f"Training set size: {X_train_scaled.shape[0]}")
        print(f"Testing set size: {X_test_scaled.shape[0]}")
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def train_model(self, X_train, y_train):
        """
        Train the Random Forest model
        
        Args:
            X_train: Training features
            y_train: Training labels
        """
        print("\nTraining Random Forest model...")
        
        # ============ EDIT THIS: Adjust model parameters if needed ============
        self.model = RandomForestClassifier(
            n_estimators=100,      # Number of trees
            max_depth=10,          # Maximum depth of trees
            min_samples_split=5,   # Minimum samples to split a node
            min_samples_leaf=2,    # Minimum samples at leaf node
            random_state=42,
            n_jobs=-1              # Use all CPU cores
        )
        # =======================================================================
        
        self.model.fit(X_train, y_train)
        print("Model training completed!")
        
        # Display feature importance
        feature_importance = pd.DataFrame({
            'feature': self.features,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nFeature Importance:")
        print(feature_importance)
    
    def evaluate_model(self, X_test, y_test):
        """
        Evaluate model performance
        
        Args:
            X_test: Test features
            y_test: Test labels
        """
        print("\nEvaluating model...")
        
        y_pred = self.model.predict(X_test)
        
        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        print(f"\nAccuracy: {accuracy * 100:.2f}%")
        
        # Classification report
        print("\nClassification Report:")
        print(classification_report(
            y_test, y_pred, 
            target_names=self.label_encoder.classes_
        ))
        
        # Confusion matrix
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
    
    def predict(self, input_data):
        """
        Make prediction on new data
        
        Args:
            input_data (dict or DataFrame): Input features
            
        Returns:
            tuple: (severity_prediction, confidence_scores)
        """
        if self.model is None:
            raise ValueError("Model not trained yet! Please train the model first.")
        
        # Convert input to DataFrame if it's a dictionary
        if isinstance(input_data, dict):
            input_df = pd.DataFrame([input_data])
        else:
            input_df = input_data
        
        # Ensure correct feature order
        input_df = input_df[self.features]
        
        # Scale input
        input_scaled = self.scaler.transform(input_df)
        
        # Make prediction
        prediction = self.model.predict(input_scaled)
        probabilities = self.model.predict_proba(input_scaled)
        
        # Decode prediction
        severity = self.label_encoder.inverse_transform(prediction)[0]
        
        # Get confidence for each class
        confidence_dict = {
            class_name: f"{prob * 100:.2f}%" 
            for class_name, prob in zip(self.label_encoder.classes_, probabilities[0])
        }
        
        return severity, confidence_dict
    
    def save_model(self, filepath='coral_model.pkl'):
        """Save trained model to file"""
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'features': self.features
        }, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath='coral_model.pkl'):
        """Load trained model from file"""
        data = joblib.load(filepath)
        self.model = data['model']
        self.scaler = data['scaler']
        self.label_encoder = data['label_encoder']
        self.features = data['features']
        print(f"Model loaded from {filepath}")


# ======================== EXAMPLE USAGE ========================

if __name__ == "__main__":
    # Initialize predictor
    predictor = CoralHealthPredictor()
    
    # ============ EDIT THIS: Specify your CSV file path ============
    CSV_FILE_PATH = "cleaned_data5.csv"  # Change this to your CSV file path
    # ================================================================
    
    # Load and train model
    df = predictor.load_data(CSV_FILE_PATH)
    X_train, X_test, y_train, y_test = predictor.preprocess_data(df)
    predictor.train_model(X_train, y_train)
    predictor.evaluate_model(X_test, y_test)
    
    # Save the trained model
    predictor.save_model('coral_model.pkl')
    
    # ============ EDIT THIS: Make predictions on new data ============
    # Example: Single prediction
    new_data = {
    'Latitude_Degrees': 15,
    'Longitude_Degrees': 130,
    'Turbidity': 0.08,
    'Cyclone_Frequency': 70,
    'Temperature_Mean': 31,
    'Windspeed': 4,
    'SSTA': 2.1,
    'SSTA_DHW': 4.5,
    'TSA': 3.2,
    'TSA_DHWMean': 1.5,
    'Exposure': 'Exposed'
        # Add values for any additional features you defined above
    }
    
    severity, confidence = predictor.predict(new_data)
    
    print("\n" + "="*50)
    print("PREDICTION RESULTS")
    print("="*50)
    print(f"Input data: {new_data}")
    print(f"\nPredicted Severity: {severity}")
    print(f"Confidence scores: {confidence}")
    print("="*50)
    
    # Example: Multiple predictions
   # print("\n\nMultiple predictions example:")
    #multiple_data = pd.DataFrame([
        #{'temperature': 29.0, 'turbidity': 3.5, 'salinity': 34.8, 'ph': 7.9, 'dissolved_oxygen': 6.2},
        #{'temperature': 26.5, 'turbidity': 1.2, 'salinity': 35.5, 'ph': 8.3, 'dissolved_oxygen': 7.5},
        #{'temperature': 31.0, 'turbidity': 5.0, 'salinity': 34.0, 'ph': 7.5, 'dissolved_oxygen': 5.0},
   # ])
    
    #for idx, row in multiple_data.iterrows():
        #severity, confidence = predictor.predict(row.to_dict())
        #print(f"\nSample {idx + 1}: Predicted Severity = {severity}")
        #print(f"Confidence: {confidence}")