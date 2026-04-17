from flask import Flask, render_template, request
import joblib
import pandas as pd
model = joblib.load('model_joblib_heart')

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():

    print(request.form)

    try:
        age = int(request.form['age'])
        sex = int(request.form['sex'])
        cp = int(request.form['cp'])
        trestbps = int(request.form['trestbps'])
        chol = int(request.form['chol'])
        fbs = int(request.form['fbs'])
        restecg = int(request.form['restecg'])
        thalach = int(request.form['thalach'])
        exang = int(request.form['exang'])
        oldpeak = float(request.form['oldpeak'])
        slope = int(request.form['slope'])
        ca = int(request.form['ca'])
        thal = int(request.form['thal'])

        input_data = pd.DataFrame([[age, sex, cp, trestbps, chol,
                                     fbs, restecg, thalach,
                                     exang, oldpeak, slope, ca, thal]],
                                   columns=['age', 'sex', 'cp', 'trestbps', 'chol',
                                            'fbs', 'restecg', 'thalach',
                                            'exang', 'oldpeak', 'slope', 'ca', 'thal'])
        prediction = model.predict(input_data)

        output = "Heart Disease" if prediction[0] == 1 else "No Heart Disease"

    except Exception as e:
        output = f"Error: {str(e)}"

    return render_template('result.html', result=output)

if __name__ == "__main__":
    app.run(debug=True)
