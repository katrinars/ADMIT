# Machine Learning Capstone Project

### PROJECT REQUIREMENTS
This open-ended project asked students to choose a mock business problem that machine learning could be applied to. 
Students were then asked to source data and use that data and their chosen machine learning algorithms to provide 
descriptive and prescriptive insights of their choice. Lastly, students were asked to create an interactive interface 
where their data could be explored. 


### KEY LEARNINGS
- Machine Learning Libraries (NumPy, Pandas, Matplotlib, Jupyter)
- Logistic Regression Implementation
- A bit of Javascript
  

### ACCESS THE APPLICATION
1. Navigate to https://alaskaadmissions.fly.dev. It may take several minutes for the site to build.
2. Enter the username 'alaska' and password 'university' to enter the admissions portal. When the dashboard is accessed the model is created and the dashboard is populated with charts based on the 
<a href="static/university_admission.csv">`sample data`</a>.


### FEATURES 
There are three interactive features in the sidebar. 

#### Upload Applicants
This form requires a CSV file with specific formatting. Use the <a href="static/demo_applicants (1).csv">`demo applicants`</a> file 
here to test the model's ability to generate new admissions decisions and update the charts based on new applicant data. Use the New Decisions 
button in the Download Data area to see the admissions decisions.

#### Likelihood of Admission
Enter Score values here to generate an individual admissions decision. The probability functionality available with logistic 
regression models is applied here as well.

#### Download Data
The Historic Data button downloads the <a href="static/university_admission.csv">`sample data`</a>. The New Decisions button downloads 
the <a href="static/demo_applicants (1).csv">`demo applicants`</a> file with the admissions decisions populated. The New Decisions button 
will not work until new data is uploaded in the Upload Applicants form.
