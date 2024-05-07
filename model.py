import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, ConfusionMatrixDisplay, RocCurveDisplay
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegressionCV

# %%
# import data and print dataframe
data = "./static/university_admission.csv"
df = pd.read_csv(data)
df

# %%
df.isnull().sum()

#%%
det = df.describe()
det

# %%
# gre boxplot
fig, ax = plt.subplots()
gre_box = sns.boxplot(x=df["gre"])
plt.title("GRE Boxplot")
plt.show()
plt.close()

# sop boxplot
sop_box = sns.boxplot(x=df["sop"])
plt.title("SOP Boxplot")
plt.show()
plt.close()

# cgpa boxplot
cgpa_box = sns.boxplot(x=df["cgpa"])
plt.title("CGPA Boxplot")
plt.show()
plt.close()

# %%
# insert applicant no.
df.insert(0, 'applicant no.', value=np.arange(1, len(df) + 1))

# %%
# save new dataframe to csv for dashboard
df.to_csv("./static/au_admissions.csv", index=False)

# %%
# group independent vs. dependent variables
X = df.drop(columns=['applicant no.', 'admitted'])
y = pd.DataFrame(df['admitted'])

# %%
# scale features
sc = StandardScaler()
X = sc.fit_transform(X.values)

# %%
# split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# %%
# train model
y_train_array, y_test_array = y_train['admitted'].values, y_test['admitted'].values
log_model = LogisticRegressionCV(cv=5, random_state=42, refit=True).fit(X_train, y_train_array)

# %%
# get probabilities
probabilities = log_model.predict_proba(X_test)
print(probabilities)

# %%
# get predictions
predictions = log_model.predict(X_test)
print(predictions)

# %%
# confusion matrix with values as percentages
cm = ConfusionMatrixDisplay.from_predictions(y_test, predictions, display_labels=['Reject', 'Admit'])
plt.show()
plt.close()

# %%
# get scores
accuracy = accuracy_score(y_test, predictions)
f1 = f1_score(y_test, predictions)
auc_score = roc_auc_score(y_test, predictions)

# print scores with two decimal places
print(f"Accuracy: {accuracy:.2f}")
print(f"F1 Score: {f1:.2f}")
print(f"AUC Score: {auc_score:.2f}")

# %%
# plot roc curve
roc_curve = RocCurveDisplay.from_predictions(y_test, predictions)
plt.show()
plt.close()
