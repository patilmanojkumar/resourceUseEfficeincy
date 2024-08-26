import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm

# Function to estimate Cobb-Douglas production function
def cobb_douglas_regression(df):
    df_log = np.log(df)
    Y = df_log.iloc[:, 0]  # First column as dependent variable
    X = df_log.iloc[:, 1:]  # All other columns as independent variables
    X = sm.add_constant(X)  # Add constant term to the model
    model = sm.OLS(Y, X).fit()
    return model, df_log

# Function to calculate Resource Use Efficiency
def calculate_mvp(df_log, model):
    # Calculate the geometric mean of each variable
    geomeans = np.exp(df_log.mean())

    mvp = {}
    for i, coef in enumerate(model.params[1:], start=1):  # Skipping the constant term
        mvp[f'MVP_{i}'] = (geomeans[0] / geomeans[i]) * coef
    return mvp

# Streamlit app starts here
st.title("Cobb-Douglas Production Function Estimation")
st.write("Upload a CSV or Excel file with the following columns in order: Output, Farm size, Seed cost, Irrigation cost, Land preparation cost, Pesticides cost, Manure cost, Chemical fertilizer cost, Labor cost.")

# File upload
uploaded_file = st.file_uploader("Upload your file", type=["csv", "xlsx"])

if uploaded_file:
    # Read the file
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    # Display the dataframe
    st.write("Data Preview:")
    st.dataframe(df)

    # Perform Cobb-Douglas regression
    model, df_log = cobb_douglas_regression(df)

    # Display the results
    st.write("### Table 3. Estimated value of coefficients and related statistics")
    results_table = pd.DataFrame({
        "Variables": ["Constant"] + df.columns[1:].tolist(),
        "Parameters": model.params.index.tolist(),
        "Co-efficient": model.params.values,
        "Standard error": model.bse.values,
        "t-ratio": model.tvalues.values
    })
    st.table(results_table)

    # Calculate and display Resource Use Efficiency
    mvp = calculate_mvp(df_log, model)
    st.write("### Table 4. Resource use efficiency in Cobb-Douglas production")
    resource_efficiency_table = pd.DataFrame({
        "Name of variables": df.columns[1:].tolist(),
        "Coefficients": model.params[1:].values,
        "MVP": list(mvp.values()),
        "MVP/MFC": list(mvp.values())  # Assuming MFC is 1 for simplicity
    })
    st.table(resource_efficiency_table)

    # Provide option to download the tables
    def convert_df(df):
        return df.to_csv(index=False).encode('utf-8')

    st.download_button(
        label="Download Table 3 as CSV",
        data=convert_df(results_table),
        file_name='cobb_douglas_coefficients.csv',
        mime='text/csv',
    )

    st.download_button(
        label="Download Table 4 as CSV",
        data=convert_df(resource_efficiency_table),
        file_name='resource_use_efficiency.csv',
        mime='text/csv',
    )
