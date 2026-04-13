from flask import Flask, render_template, request
import joblib
import pandas as pd
import os

app = Flask(__name__)

model = joblib.load('model.pkl')
features = joblib.load('features.pkl')

file_name = 'house_price_regression_dataset.csv'
df = pd.read_csv(file_name)
df_sample = df.head(10)
-
@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        try:
            inputs = [float(request.form[f]) for f in features]
            pred_value = model.predict([inputs])[0]
            prediction = f"${pred_value:,.2f}"
        except Exception as e:
            prediction = f"Error: {e}"

    return render_template('index.html', 
                           features=features, 
                           prediction=prediction,
                           table=df_sample.to_html(classes='table table-dark table-striped'))

if __name__ == '__main__':
    if not os.path.exists('static'):
        os.makedirs('static')
    app.run(debug=True)