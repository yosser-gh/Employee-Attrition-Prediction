#Load the required Python libraries and the dataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px

# Load the dataset
df = pd.read_csv('./Data/Employee-Attrition.csv')

"""**Data exploration**"""

df.columns

# Checking for missing values
print(df.isnull().sum())

"""**Data visualization**"""
"""
sns.displot(df['Age'], kde=True)
plt.title('Distribution of Age')
plt.show()

df.groupby(['Age', 'Attrition']).size().unstack().plot(kind='bar', stacked=True, figsize=(12, 8))


# Filter the data to show only "Yes" values in the "Attrition" column
attrition_data = df[df['Attrition'] == 'Yes']

# Calculate the count of attrition by department
attrition_by = attrition_data.groupby(['Department']).size().reset_index(name='Count')

# Create a donut chart
fig = go.Figure(data=[go.Pie(
    labels=attrition_by['Department'],
    values=attrition_by['Count'],
    hole=0.4,
    marker=dict(colors=['#3CAEA3', '#F6D55C']),
    textposition='inside'
)])

# Update the layout
fig.update_layout(title='Attrition by Department', font=dict(size=16), legend=dict(
    orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
))

# Show the chart
fig.show()

attrition_by = attrition_data.groupby(['EducationField']).size().reset_index(name='Count')

# Create a donut chart
fig = go.Figure(data=[go.Pie(
    labels=attrition_by['EducationField'],
    values=attrition_by['Count'],
    hole=0.4,
    marker=dict(colors=['#3CAEA3', '#F6D55C']),
    textposition='inside'
)])

# Update the layout
fig.update_layout(title='Attrition by Educational Field', font=dict(size=14), legend=dict(
    orientation="h", yanchor="bottom", y=1, xanchor="right", x=1
))


# Show the chart
fig.show()

attrition_by = attrition_data.groupby(['Gender']).size().reset_index(name='Count')

# Create a donut chart
fig = go.Figure(data=[go.Pie(
    labels=attrition_by['Gender'],
    values=attrition_by['Count'],
    hole=0.4,
    marker=dict(colors=['#3CAEA3', '#F6D55C']),
    textposition='inside'
)])

# Update the layout
fig.update_layout(title='Attrition by Gender', font=dict(size=16), legend=dict(
    orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
))

# Show the chart
fig.show()

fig = px.scatter(df, x="Age", y="MonthlyIncome", color="Attrition", trendline="ols")
fig.update_layout(title="Age vs. Monthly Income by Attrition")

fig.show()
"""
def encode(df):
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()

    label_encoders = {}
    binary_columns = ['Attrition', 'OverTime', 'Gender','Over18']

    for column in binary_columns:
        if column in df.columns:
            le = LabelEncoder()
            df[column] = le.fit_transform(df[column])
            label_encoders[column] = le

    # One-hot encode categorical columns
    for column in ['JobRole', 'Department', 'MaritalStatus','EducationField','BusinessTravel']:
        if column in df.columns:
            df= pd.get_dummies(df, columns=[column], drop_first=True)
    return df

# Select only numeric columns

"""
# Plot the heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Matrix of Numeric Columns')
plt.show()
"""
"""**Normalize data**"""

# Normalize numerical columns
def normalize(df):
    from sklearn.discriminant_analysis import StandardScaler
    numeric_df = ['Age','MonthlyIncome','TotalWorkingYears','TrainingTimesLastYear','YearsAtCompany','YearsInCurrentRole', 'YearsSinceLastPromotion',
       'YearsWithCurrManager', ]
    for column in numeric_df:
        if column in df.columns:
            scaler = StandardScaler()
            df[column] = scaler.fit_transform(df[[column]])
    return df
