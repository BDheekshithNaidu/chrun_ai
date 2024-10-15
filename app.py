import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from keras.models import load_model
from sklearn.preprocessing import StandardScaler
import joblib
import plotly.express as px
# Load the saved best model and scaler
model = load_model('best_demo_customer_churn_model_nn.h5')
scaler = joblib.load('scaler.pkl')  # Assuming you've saved the scaler as 'scaler.pkl'

np.random.seed(42)
data = {
    "CustomerID": range(1, 101),
    "ServiceQuality": np.random.randint(1, 11, size=100),  # 1 to 10
    "PricingAndPlans": np.random.choice(['Basic', 'Standard', 'Premium'], size=100),
    "ContractualObligations": np.random.choice(['None', '1 Year', '2 Years'], size=100),
    "CompetitorOffers": np.random.randint(0, 2, size=100),  # 0 or 1
    "CustomerEngagement": np.random.randint(1, 6, size=100),  # 1 to 5
    "BillingIssues": np.random.randint(0, 2, size=100),  # 0 or 1
    "ServiceChanges": np.random.randint(0, 2, size=100),  # 0 or 1
    "CustomerLifecycleStage": np.random.choice(['New', 'Active', 'At-Risk', 'Churned'], size=100),
    "UsagePatterns": np.random.randint(1, 101, size=100),  # 1 to 100
    "EconomicFactors": np.random.randint(1, 11, size=100),  # 1 to 10
    "TechnologyTrends": np.random.randint(1, 11, size=100),  # 1 to 10
    "CustomerSentiment": np.random.randint(1, 6, size=100),  # 1 to 5
}

df = pd.DataFrame(data)

# Refine churn based on additional conditions
conditions = [
    (df['ServiceQuality'] <= 4) & (df['CustomerEngagement'] <= 2) & (df['BillingIssues'] == 1),  # Poor service, low engagement, billing issues
    (df['CompetitorOffers'] == 1) & (df['ContractualObligations'] == 'None'),  # Competitor offer and no contract
    (df['CustomerLifecycleStage'].isin(['At-Risk', 'Churned'])),  # Already at-risk or churned customers
    (df['PricingAndPlans'] == 'Basic') & (df['ServiceQuality'] <= 5),  # Basic plan and low service quality
    (df['EconomicFactors'] >= 8) & (df['ServiceQuality'] <= 5),  # Economic stress and low service quality
    (df['UsagePatterns'] <= 20),  # Very low usage
    (df['CustomerSentiment'] <= 2),  # Low sentiment reflects dissatisfaction
]

# Assign churn based on conditions
df['Churn'] = np.select(conditions, [1, 1, 1, 1, 1, 1, 1], default=0)

# df = pd.read_csv("inline_read_data.csv")


# Display Churn Prediction App Title
st.title('Churn Prediction App')

# Section: Data Visualization
st.write("## Churn Insights Visualization")

# Visualizing churn distribution
fig_pie = px.pie(df, names='Churn', title='Churn Distribution')
st.plotly_chart(fig_pie)

# Service quality vs churn (useful to see how service quality affects churn)
fig_bar = px.bar(df, x='ServiceQuality', y='Churn', title='Service Quality vs Churn',
                 color='Churn', barmode='group')
st.plotly_chart(fig_bar)

# Adding more features for visualization
fig_scatter = px.scatter(df, x='UsagePatterns', y='CustomerSentiment',
                         color='Churn', title="Customer Sentiment vs Usage Patterns")
st.plotly_chart(fig_scatter)

# Move the form to the sidebar
st.sidebar.write("## Enter Customer Data for Prediction")
with st.sidebar.form("churn_prediction_form"):
    service_quality = st.slider("Service Quality", 1, 10, 5)
    pricing_and_plans = st.selectbox("Pricing and Plans", ['Basic', 'Standard', 'Premium'])
    contractual_obligations = st.selectbox("Contractual Obligations", ['None', '1 Year', '2 Years'])
    competitor_offers = st.selectbox("Competitor Offers", [0, 1])
    customer_engagement = st.slider("Customer Engagement", 1, 5, 3)
    billing_issues = st.selectbox("Billing Issues", [0, 1])
    service_changes = st.selectbox("Service Changes", [0, 1])
    customer_lifecycle_stage = st.selectbox("Customer Lifecycle Stage", ['New', 'Active', 'At-Risk', 'Churned'])
    usage_patterns = st.slider("Usage Patterns", 1, 100, 50)
    economic_factors = st.slider("Economic Factors", 1, 10, 5)
    technology_trends = st.slider("Technology Trends", 1, 10, 5)
    customer_sentiment = st.slider("Customer Sentiment", 1, 5, 3)

    # Submit button
    submit_button = st.form_submit_button(label="Predict Churn")

# Handle categorical features encoding (same as during training)
def encode_features(pricing_and_plans, contractual_obligations, customer_lifecycle_stage):
    # Encode 'PricingAndPlans'
    pricing_map = {'Basic': [0, 0], 'Standard': [1, 0], 'Premium': [0, 1]}
    contractual_map = {'None': [0, 0], '1 Year': [1, 0], '2 Years': [0, 1]}
    lifecycle_map = {'New': [0, 0, 0], 'Active': [1, 0, 0], 'At-Risk': [0, 1, 0], 'Churned': [0, 0, 1]}
    
    encoded_pricing = pricing_map[pricing_and_plans]
    encoded_contract = contractual_map[contractual_obligations]
    encoded_lifecycle = lifecycle_map[customer_lifecycle_stage]

    return encoded_pricing + encoded_contract + encoded_lifecycle

if submit_button:
    # Encode categorical features
    encoded_features = encode_features(pricing_and_plans, contractual_obligations, customer_lifecycle_stage)

    # Preprocess the input data for prediction
    new_data = np.array([[service_quality, competitor_offers, customer_engagement,
                          billing_issues, service_changes, usage_patterns,
                          economic_factors, technology_trends, customer_sentiment] + encoded_features])

    # Apply scaling to the new data
    new_data_scaled = scaler.transform(new_data)
    
    # Perform prediction
    prediction = model.predict(new_data_scaled)
    churn_probability = prediction[0][0]
    
    # Display the prediction result
    st.sidebar.write(f"### Churn Probability: {churn_probability:.4f}")
    if churn_probability > 0.5:
        st.sidebar.warning("The customer is likely to churn.")
    else:
        st.sidebar.success("The customer is unlikely to churn.")
