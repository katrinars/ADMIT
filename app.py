import jsonpickle
from flask import Flask, render_template, request
from sklearn.preprocessing import StandardScaler

import model
import pandas as pd




app = Flask(__name__)




@app.route('/', methods=['GET', 'POST'])
def login():
    # https://realpython.com/introduction-to-flask-part-2-creating-a-login-page/
    error = None
    if request.method == 'POST':
        if request.form['username'] != 'alaska' or request.form['password'] != 'university':
            error = 'Invalid Credentials. Please try again.'
        else:
            return render_template('dashboard.html')
    return render_template('index.html', error=error)


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Get the values from the form
        gre = float(request.form.get('gre'))
        cgpa = float(request.form.get('cgpa'))
        sop = float(request.form.get('sop'))

        # Use the model to make a prediction
        prediction = model.log_model.predict([[gre, cgpa, sop]])
        if prediction[0] == 0:
            prediction = 'Reject'
        else:
            prediction = 'Admit'
        probability = model.log_model.predict_proba([[gre, cgpa, sop]])

        # Return the prediction
        return f'DECISION = {prediction} - Rating: {probability[0][1] * 100:.2f}% chance'
    else:
        # Render the form
        return render_template('predict.html')



@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from the request
        file = request.files['applicants']

        # create dataframe and segment variables
        new_data = pd.read_csv(file)
        X = new_data.drop(columns=['applicant no.'])

        # scale independent variables
        X = pd.DataFrame(model.sc.transform(X))

        # apply logistic regression model and insert decisions into dataframe
        decisions = model.log_model.predict(X)
        new_data.insert(4, 'decision', decisions)
        results = {new_data.to_html(index=False)}

        # Return a success message
        return jsonpickle.encode(results)
    else:
        # Render the form
        return render_template('upload.html')


if __name__ == '__main__':
    app.run(debug=True)
