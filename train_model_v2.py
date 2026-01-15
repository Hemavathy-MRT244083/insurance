# train_improved_model.py
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# 1. Load dataset
df = pd.read_csv("C:/Users/HEMA/C Program/insurance/data/insurance.csv")

# 2. Features and target
X = df.drop("charges", axis=1)
y = df["charges"]

categorical_features = ["sex", "smoker", "region"]
numerical_features = ["age", "bmi", "children"]

# 3. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Preprocessing with improved pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ("num", Pipeline([
            ("scaler", StandardScaler()),
            ("poly", PolynomialFeatures(degree=2, include_bias=False))
        ]), numerical_features),
        ("cat", OneHotEncoder(drop='first'), categorical_features)
    ]
)

# 5. Create pipeline with regression
improved_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", LinearRegression())
])

# 6. Train improved model
improved_pipeline.fit(X_train, y_train)

# 7. Predict & Evaluate
y_pred = improved_pipeline.predict(X_test)
rmse = mean_squared_error(y_test, y_pred) ** 0.5
r2 = r2_score(y_test, y_pred)

print(f"Improved Model (v2) Performance:\nRMSE: {rmse:.2f}\nR²: {r2:.2f}")

# 8. Save model
joblib.dump(improved_pipeline, "model_v2.pkl")
print("Improved model saved as model_v2.pkl")
