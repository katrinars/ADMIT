# import libraries
import pretty_jupyter
import pandas as pd
import seaborn as sns
from app import *

sns.set_theme()

# %%
# import data and print raw dataframe
data = "static/au_admissions.csv"
df = pd.read_csv(data)

print(df.min())


# %%
# get scores for specific applicant
def get_prediction(df, applicant):
    # locate applicant index in dataframe and print scores
    applicant_gre = df.iloc[applicant - 1]['gre']
    applicant_cgpa = df.iloc[applicant - 1]['cgpa']
    applicant_sop = df.iloc[applicant - 1]['sop']
    print(f' APPLICANT {applicant} - GRE: {applicant_gre}, CGPA: {applicant_cgpa}, SOP: {applicant_sop}')

print(get_prediction(df, 1))
