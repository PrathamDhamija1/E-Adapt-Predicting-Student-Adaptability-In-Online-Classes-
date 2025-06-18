# 1. Import Necessary Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib

# 2. Read the Dataset
df = pd.read_csv('saloe.csv')

# 3. Preprocessing
# 3.1 Encoding Categorical Variables
label_encoder = LabelEncoder()
for column in df.select_dtypes(include='object').columns:
    df[column] = label_encoder.fit_transform(df[column])

# 3.2 Train-Test Split
X = df.drop('Adaptivity Level', axis=1)
y = df['Adaptivity Level']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=116)

# 3.3 Scale the Data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. Train the RandomForest Model
model = RandomForestClassifier(random_state=116)
model.fit(X_train_scaled, y_train)

# 5. Save the Model and Scaler
joblib.dump(model, 'random_forest_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
print("Model and scaler saved as 'random_forest_model.pkl' and 'scaler.pkl'")