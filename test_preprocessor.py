"""
Test if the saved preprocessor is fitted
"""
import joblib
import pandas as pd

print("Loading preprocessor...")
preprocessor = joblib.load("preprocessor.pkl")

print(f"Preprocessor type: {type(preprocessor)}")
print(f"Preprocessor fitted: {hasattr(preprocessor, '_fitted') or hasattr(preprocessor, 'transformers_')}")

# Check if it has the transformers_ attribute (only present after fitting)
if hasattr(preprocessor, 'transformers_'):
    print("SUCCESS: Preprocessor is FITTED")
    print(f"Number of transformers: {len(preprocessor.transformers_)}")

    # Check the categorical transformer
    for name, transformer, cols in preprocessor.transformers_:
        print(f"\nTransformer '{name}' for columns: {cols}")
        if name == "cat":
            if hasattr(transformer.named_steps['onehot'], 'categories_'):
                print(f"  OneHotEncoder categories: {transformer.named_steps['onehot'].categories_}")
            else:
                print("  WARNING: OneHotEncoder is NOT fitted!")
else:
    print("ERROR: Preprocessor is NOT fitted")

# Test with sample data
print("\nTesting transformation...")
test_data = pd.DataFrame([{
    "fixed acidity": 7.4,
    "volatile acidity": 0.7,
    "citric acid": 0.0,
    "residual sugar": 1.9,
    "chlorides": 0.076,
    "free sulfur dioxide": 11.0,
    "total sulfur dioxide": 34.0,
    "density": 0.9978,
    "pH": 3.51,
    "sulphates": 0.56,
    "alcohol": 9.4,
    "sulfur_ratio": 0.323,
    "color": "red"
}])

try:
    result = preprocessor.transform(test_data)
    print(f"SUCCESS: Transformation successful! Output shape: {result.shape}")
except Exception as e:
    print(f"ERROR: Transformation failed: {e}")
