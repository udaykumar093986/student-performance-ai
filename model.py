import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import pickle

# Load dataset
_df = pd.read_csv("student_data.csv")

# Save original columns
label_encoders = {}

# Encode categorical columns
for col in _df.columns:
    if _df[col].dtype == 'object':
        le = LabelEncoder()
        _df[col] = le.fit_transform(_df[col])
        label_encoders[col] = le

# Features and target
X = _df.drop("math score", axis=1)
y = _df["math score"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train model
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# Save model
pickle.dump(model, open("model.pkl", "wb"))

print("Model trained successfully")