import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib

# Load the dataset
data = pd.read_csv('C:\\Users\\kiran\\Desktop\\RF PROJECT\\banking crisis_encoded_scaled.csv')

# Initialize LabelEncoder
label_encoder = LabelEncoder()

# Apply label encoding to 'cc3', 'country', and 'banking_crisis' columns
data['cc3'] = label_encoder.fit_transform(data['cc3'])
data['country'] = label_encoder.fit_transform(data['country'])
data['banking_crisis'] = label_encoder.fit_transform(data['banking_crisis'])

# Save the label encoder for future use in the Flask app
joblib.dump(label_encoder, 'label_encoder.pkl')

# Select relevant features and target variable
features = ['systemic_crisis', 'year', 'inflation_annual_cpi', 'exch_usd']
target = 'banking_crisis'

X = data[features]
y = data[target]

# Initialize MinMaxScaler
scaler = MinMaxScaler()

# Apply Min-Max scaling to 'exch_usd' and 'inflation_annual_cpi' columns
X[['exch_usd', 'inflation_annual_cpi']] = scaler.fit_transform(X[['exch_usd', 'inflation_annual_cpi']])

# Save the scaler for future use in the Flask app
joblib.dump(scaler, 'scaler.pkl')

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize and train the RandomForestClassifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save the trained model for future use in the Flask app
joblib.dump(model, 'banking_crisis_model.pkl')

# Evaluate the model
y_pred = model.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print('Accuracy:', accuracy_score(y_test, y_pred))
