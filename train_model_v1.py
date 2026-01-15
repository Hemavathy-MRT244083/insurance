import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# 1. Load dataset
df = pd.read_csv("C:/Users/HEMA/C Program/insurance/data/insurance.csv")

# 2. Features & target
X = df.drop("charges", axis=1)
y = df["charges"]

# Identify categorical and numerical columns
categorical_features = ["sex", "smoker", "region"]
numerical_features = ["age", "bmi", "children"]

# 3. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_features),
        ("cat", OneHotEncoder(drop='first'), categorical_features)
    ]
)

# 5. Create pipeline
pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", LinearRegression())
])

# 6. Train baseline model
pipeline.fit(X_train, y_train)

# 7. Make predictions
y_pred = pipeline.predict(X_test)

# 8. Evaluate model
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
r2 = r2_score(y_test, y_pred)

print(f"Baseline Model (v1) Performance:\nRMSE: {rmse:.2f}\nR^2: {r2:.2f}")

# 9. Save the trained baseline model
joblib.dump(pipeline, "model_v1.pkl")
print("Baseline model saved as model_v1.pkl")
