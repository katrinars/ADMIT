from flask import Flask, render_template, request, redirect, url_for

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
