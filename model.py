# import libraries
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.model_selection import KFold, cross_val_score

# %%
# import data and print raw dataframe
url = "./static/au_admissions.csv"
df = pd.read_csv(url)

# %%
# split data into independent and dependent variables
X = df.drop(columns=['applicant no.', 'decision'])
y = pd.DataFrame(df['decision'])

# %%
# split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# %%
# feature scaling
sc = StandardScaler()
X_train = pd.DataFrame(sc.fit_transform(X_train))
X_test = pd.DataFrame(sc.transform(X_test))

# %%
# apply logistic regression model
y_train_array, y_test_array = y_train['decision'].values, y_test['decision'].values
log_model = LogisticRegression(random_state=42).fit(X_train, y_train_array)

# %%
# predictions log and accuracy score
predictions_log = log_model.predict(X_test)
print(predictions_log)
score = metrics.accuracy_score(y_test, predictions_log)
print("Accuracy Score: ", score)

# %%
f1_score = metrics.f1_score(y_test, predictions_log)
print("F1 Score: ", f1_score)

# %%
# apply k-folds cross validation
y_array = df['decision'].values
k_folds = KFold(n_splits=5, shuffle=True)
scores = cross_val_score(log_model, X, y_array)
print("Split Scores: ", scores)
print("Average Score: ", scores.mean())

# %%
auc_score = metrics.roc_auc_score(y_test, predictions_log)
print("AUC Score: ", auc_score)
