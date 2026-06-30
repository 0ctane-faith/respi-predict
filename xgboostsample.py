
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# 1. Simulate a Patient Dataset (Replace this with your Kaggle CSV)

# Features: Age, Oxygen Saturation (SpO2), Heart Rate, Smoker (1=Yes, 0=No)

# Target: Disease State (0 = Healthy, 1 = Asthma, 2 = COPD)

data = {
    'Age': np.random.randint(18, 80, 500),
    'SpO2': np.random.uniform(85, 100, 500),
    'HeartRate': np.random.randint(60, 110, 500),
    'Smoker': np.random.randint(0, 2, 500),
    'Target_Disease': np.random.randint(0, 3, 500) 
}
df = pd.DataFrame(data)

# 2. Split the Data into Features (X) and Target (y)
X = df.drop('Target_Disease', axis=1)
y = df['Target_Disease']

# Split into Training (80%) and Testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Initialize and Train the XGBoost Model
print("Training the XGBoost Prediction Model...")
model = xgb.XGBClassifier(objective='multi:softprob', num_class=3, eval_metric='mlogloss')
model.fit(X_train, y_train)

# 4. Make Predictions on the unseen Test data
predictions = model.predict(X_test)

# 5. Evaluate the Results
print("\n--- Model Evaluation ---")
print(f"Accuracy: {accuracy_score(y_test, predictions) * 100:.2f}%")
print("\nDetailed Report:")
print(classification_report(y_test, predictions, target_names=['Healthy', 'Asthma', 'COPD']))
