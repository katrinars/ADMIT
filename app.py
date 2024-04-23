import sklearn
from flask import Flask, render_template, request


import model
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

matplotlib.use('Agg')

app = Flask(__name__)
sns.set_theme(palette='Pastel1', context='talk', font='sans-serif', font_scale=0.8, style='dark')


@app.route('/', methods=['GET', 'POST'])
def login():
    # https://realpython.com/introduction-to-flask-part-2-creating-a-login-page/
    error = None
    if request.method == 'POST':
        # display dashboard if login is valid, else display error message
        if request.form['username'] != 'alaska' or request.form['password'] != 'university':
            error = 'Invalid Credentials. Please try again. \n Hint: all lowercase'
        else:
            dashboard()
            return render_template('dashboard.html')
    return render_template('index.html', error=error)


@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # retrieve uploaded file, create dataframe, and scale ind. variables
        file = request.files['applicants']
        new_data = pd.read_csv(file)
        X = new_data.drop(columns=['applicant no.'])
        X = pd.DataFrame(model.sc.transform(X.values))

        # apply ml model and insert decisions into dataframe
        decisions = model.log_model.predict(X)
        new_data.insert(4, 'decision', decisions)

        new_data.to_csv('static/new_data.csv', index=False)

        # regenerate charts using new data
        scatterplot(new_data)
        decision_pie(new_data)
        gre_hist(new_data)
        cgpa_hist(new_data)
        sop_hist(new_data)
        correlation_matrix(new_data)

        return render_template('upload.html')

    else:
        return render_template('upload.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # retrieve and scale scores from the form
        gre = float(request.form.get('gre'))
        cgpa = float(request.form.get('cgpa'))
        sop = float(request.form.get('sop'))
        scores = (gre, cgpa, sop)
        sc_scores = model.sc.transform([scores])

        # generate prediction and probability using ml model
        prediction = model.log_model.predict(sc_scores)
        if prediction[0] == 0:
            prediction = 'Reject'
        else:
            prediction = 'Admit'
        probability = model.log_model.predict_proba(sc_scores)


        return (f'{probability[0][1] * 100:.2f}% Chance of Admission'
                f'   '
                f'Recommendation: {prediction}')

    else:
        return render_template('predict.html')


@app.route('/scatterplot', methods=['GET', 'POST'])
def scatterplot(df):
    # set figure size, return decisions as string
    fig, ax = plt.subplots(figsize=(4, 4), linewidth=3, edgecolor='gray')
    for decision in df['decision']:
        if decision == 1:
            decision = 'admit'
        else:
            decision = 'reject'

    # create scatterplot matrix and customize legend
    ax = sns.pairplot(df.drop(columns='applicant no.'), height=3, kind="scatter", hue='decision', corner=True, )
    ax.legend.remove()
    plt.legend(title='Decision', loc='upper right', bbox_to_anchor=(0.7, 2.3), labels=['admit', 'reject'], fontsize='x-large', facecolor='w', edgecolor='gray')

    # save plot and close
    plt.savefig('static/scatterplot.png', transparent=True)
    plt.draw()
    plt.close()

    return render_template('dashboard.html')


@app.route('/correlation', methods=['GET', 'POST'])
def correlation_matrix(df):
    # set figure size, remove applicant number, and create correlation matrix
    f, ax = plt.subplots(figsize=(6, 4))
    scores = df.drop(columns=['applicant no.'])
    sns.heatmap(scores.corr(), annot=True, cmap='Pastel1', fmt=".2f", linewidths=.7)

    # set plot layout, save, and close
    plt.tight_layout()
    plt.savefig('static/correlation.png', transparent=True)
    plt.draw()
    plt.close()


    return render_template('dashboard.html')


@app.route('/decision_pie', methods=['GET', 'POST'])
def decision_pie(df):
    # https://matplotlib.org/stable/gallery/pie_and_polar_charts/pie_and_donut_labels.html#sphx-glr-gallery-pie-and-polar-charts-pie-and-donut-labels-py

    # set figure size and group data by decision string
    fig, ax = plt.subplots(figsize=(4, 3), subplot_kw=dict(aspect="equal"))
    decision_percent = pd.DataFrame(df.groupby(by='decision').size() / df['decision'].size) * 100
    decision_percent.rename(columns={0: 'percent'}, inplace=True)
    decision = [decision for decision in decision_percent.index]
    decision[0] = 'reject'
    decision[1] = 'admit'

    # create pie chart with percentages and labels
    wedges, texts = ax.pie(decision_percent['percent'], wedgeprops=dict(width=0.4, edgecolor='white'),
                           startangle=-24)

    # add and position legend
    ax.legend(wedges, decision,
              fontsize='small',
              loc="lower left",
              bbox_to_anchor=(.6, .1, 0.2, .4))

    # add wedge annotations
    bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="gray", lw=0.5, alpha=0.4)
    kw = dict(arrowprops=dict(arrowstyle="-", fc="w", ec="gray", lw=.7),
              bbox=bbox_props, zorder=0, va="center")

    for i, p in enumerate(wedges):
        ang = (p.theta2 - p.theta1) / 2 + p.theta1
        y = np.sin(np.deg2rad(ang))
        x = np.cos(np.deg2rad(ang))
        horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x))]
        connectionstyle = f"angle,angleA=0,angleB={ang}"
        kw["arrowprops"].update({"connectionstyle": connectionstyle})
        percent = decision_percent['percent'][i]
        ax.annotate(f'{percent:.1f}%', xy=(x, y), xytext=(1.15 * np.sign(x), 1.2 * y),
                    horizontalalignment=horizontalalignment, fontsize='small', **kw)

    # set plot layout, save, and close
    plt.tight_layout()
    plt.savefig('static/decision_pie.png', transparent=True)
    plt.draw()
    plt.close()

    return render_template('dashboard.html', pie_url='static/decision_pie.png')


@app.route('/gre_hist', methods=['GET', 'POST'])
def gre_hist(df):
    # https://python-graph-gallery.com/density-mirror/

    # set figure size and group data by decision
    fig, ax = plt.subplots(figsize=(4, 4))
    top = df[df['decision'] == 1]['gre']
    bottom = df[df['decision'] == 0]['gre'] * -1

    # create top histogram for admitted students
    sns.histplot(x=top, stat="density", bins=15, edgecolor='gray')

    # get density of rejected bins and mirror
    n_bins = 15
    heights, bins = np.histogram(bottom, bins=n_bins, density=True)
    heights *= -1
    bin_width = np.diff(bins)[0]
    bin_pos = (bins[:-1] + bin_width / 2) * -1

    # create bottom histogram for rejected students
    plt.bar(bin_pos, heights, width=bin_width, edgecolor='gray')

    # set plot layout, save, and close
    plt.tight_layout(pad=1.0, w_pad=1.0, h_pad=1.0)
    plt.savefig('static/gre_hist.png', transparent=True)
    plt.draw()
    plt.close()

    return render_template('dashboard.html', gre_hist='static/gre_hist.png')


@app.route('/cgpa_hist', methods=['GET', 'POST'])
def cgpa_hist(df):
    # https://python-graph-gallery.com/density-mirror/

    fig, ax = plt.subplots(figsize=(4, 4))
    top = df[df['decision'] == 1]['cgpa']
    bottom = df[df['decision'] == 0]['cgpa'] * -1

    sns.histplot(x=top, stat="density", bins=10, edgecolor='gray')
    n_bins = 10
    heights, bins = np.histogram(bottom, density=True, bins=n_bins)
    heights *= -1
    bin_width = np.diff(bins)[0]
    bin_pos = (bins[:-1] + bin_width / 2) * -1
    plt.bar(bin_pos, heights, width=bin_width, edgecolor='gray')

    plt.tight_layout(pad=1.0, w_pad=1.0, h_pad=1.0)
    plt.savefig('static/cgpa_hist.png', transparent=True)
    plt.draw()
    plt.close()

    return render_template('dashboard.html', cgpa_hist='static/cgpa_hist.png')


@app.route('/sop_hist', methods=['GET', 'POST'])
def sop_hist(df):
    # https://python-graph-gallery.com/density-mirror/


    fig, ax = plt.subplots(figsize=(4, 4))
    top = df[df['decision'] == 1]['sop']
    bottom = df[df['decision'] == 0]['sop'] * -1

    sns.histplot(x=top, stat="density", bins=10, edgecolor='gray')
    n_bins = 10
    heights, bins = np.histogram(bottom, density=True, bins=n_bins)
    heights *= -1
    bin_width = np.diff(bins)[0]
    bin_pos = (bins[:-1] + bin_width / 2) * -1
    plt.bar(bin_pos, heights, width=bin_width, edgecolor='gray')

    plt.tight_layout(pad=1.0, w_pad=1.0, h_pad=1.0)
    plt.savefig('static/sop_hist.png', transparent=True)
    plt.draw()
    plt.close()

    return render_template('dashboard.html', sop_hist='static/sop_hist.png')


@app.route('/dashboard', methods=['GET', 'POST'])
def dashboard():
    # load new data if available, otherwise use historic data
    try:
        df = pd.read_csv('static/new_data.csv')
    except FileNotFoundError:
        df = pd.read_csv('static/au_admissions.csv')

    # refresh dashboard visuals using data found
    scatter = scatterplot(df)
    pie = decision_pie(df)
    gre_dist = gre_hist(df)
    cgpa_dist = cgpa_hist(df)
    sop_dist = sop_hist(df)
    correlation = correlation_matrix(df)
    matplotlib.pyplot.close('all')
    plt.close('all')

    return render_template('dashboard.html', scatterplot=scatter, pie=pie, gre_hist=gre_dist, cgpa_hist=cgpa_dist,
                           sop_hist=sop_dist, correlation=correlation)



@app.route('/sidebar', methods=['GET', 'POST'])
def sidebar():
    return render_template('dashboard.html')


@app.route('/charts', methods=['GET', 'POST'])
def charts():
    return render_template('dashboard.html')


if __name__ == '__main__':
    app.run(debug=True)
