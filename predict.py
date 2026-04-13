import joblib
import numpy as np

model = joblib.load('model.pkl')
features = joblib.load('features.pkl')

print("--- House Price Prediction Terminal ---")

input_data = []
for feature in features:
    while True:
        val = input(f"Enter value for {feature}: ")
        try:
            input_data.append(float(val))
            break
        except ValueError:
            print(f"Error: Please enter a valid NUMBER for {feature}.")

prediction = model.predict([input_data])
print(f"\nPredicted House Price: ${prediction[0]:,.2f}")
