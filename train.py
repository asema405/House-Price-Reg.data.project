import os
if not os.path.exists('static'):
    os.makedirs('static')

import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

file_name = 'house_price_regression_dataset.csv'
df = pd.read_csv(file_name)

print("Колонки в твоем файле:", df.columns.tolist())

possible_targets = ['Price', 'SalePrice', 'target', 'MedHouseVal', 'House_Price']
target_col = None

for col in possible_targets:
    if col in df.columns:
        target_col = col
        break

if not target_col:
    target_col = df.columns[-1]

print(f"Используем колонку '{target_col}' как целевую переменную (Y)")

df_numeric = df.select_dtypes(include=['float64', 'int64'])
X = df_numeric.drop(target_col, axis=1)
y = df_numeric[target_col]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)
accuracy = r2_score(y_test, predictions)
print(f"Точность модели (R2): {accuracy:.2%}")

joblib.dump(model, 'model.pkl')
joblib.dump(X.columns.tolist(), 'features.pkl')

plt.figure(figsize=(10, 6))
plt.scatter(y_test, predictions, alpha=0.5, color='teal')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.title(f'Actual vs Predicted ({target_col})')
plt.savefig('static/plot.png')
print("Модель и график сохранены!")
plt.show()