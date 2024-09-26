import streamlit as st
import pandas as pd
import pickle

# Load the trained Lasso model
filename = 'lasso_model.pkl'
loaded_model = pickle.load(open(filename, 'rb'))

# Create a Streamlit app
st.title('Monthly Revenue Prediction')

# Get user input for features
st.sidebar.header('Input Features')
total_orders = st.sidebar.number_input('Total Orders')
average_order_value = st.sidebar.number_input('Average Order Value')
customer_acquisition_cost = st.sidebar.number_input('Customer Acquisition Cost')
marketing_spend = st.sidebar.number_input('Marketing Spend')
website_traffic = st.sidebar.number_input('Website Traffic')
conversion_rate = st.sidebar.number_input('Conversion Rate')

# Create a feature vector from user input
input_data = pd.DataFrame({
    'total_orders': [total_orders],
    'average_order_value': [average_order_value],
    'customer_acquisition_cost': [customer_acquisition_cost],
    'marketing_spend': [marketing_spend],
    'website_traffic': [website_traffic],
    'conversion_rate': [conversion_rate]
})

# Make a prediction using the loaded model
if st.button('Predict'):
    prediction = loaded_model.predict(input_data)[0]
    st.write(f'Predicted Monthly Revenue: {prediction}')
