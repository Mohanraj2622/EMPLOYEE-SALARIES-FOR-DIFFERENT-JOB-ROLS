import pandas as pd
import numpy as np
import pandasql as ps

# Loading data
salaries = pd.read_csv('salaries.csv')

# Dropping unnecessary column
salaries = salaries.drop('Unnamed: 0', axis = 1)

# Checking for missing values
print(salaries.isna().sum())

# Converting country codes to country names
import country_converter as coco
salaries["employee_residence"] = coco.convert(names=salaries["employee_residence"], to="name")
salaries["company_location"] = coco.convert(names=salaries["company_location"], to="name")

# Converting remote ratio to a more meaningful format
salaries['remote_ratio'] = salaries['remote_ratio'].replace({'100': 'Fully Remote', '50': 'Partially Remote', '0': 'Non-Remote'})

# Grouping salaries by company location and calculating the sum of salaries
salaries_by_location = salaries.groupby('company_location')['salary_in_usd'].sum().reset_index()

# Calculating the average salary for each job title
avg_salary_by_job_title = salaries.groupby('job_title')['salary_in_usd'].mean().reset_index()
