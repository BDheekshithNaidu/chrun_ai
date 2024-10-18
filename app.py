import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from keras.models import load_model
import joblib
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

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
    (df['ServiceQuality'] <= 4) & (df['CustomerEngagement'] <= 2) & (df['BillingIssues'] == 1),
    (df['CompetitorOffers'] == 1) & (df['ContractualObligations'] == 'None'),
    (df['CustomerLifecycleStage'].isin(['At-Risk', 'Churned'])),
    (df['PricingAndPlans'] == 'Basic') & (df['ServiceQuality'] <= 5),
    (df['EconomicFactors'] >= 8) & (df['ServiceQuality'] <= 5),
    (df['UsagePatterns'] <= 20),
    (df['CustomerSentiment'] <= 2),
]

df['Churn'] = np.select(conditions, [1, 1, 1, 1, 1, 1, 1], default=0)

# Display Churn Prediction App Title
st.title('Churn Prediction App')

# Display Churned Customers Data
st.write("## Churned Customers Data")
churned_customers = df[df['Churn'] == 1]
st.dataframe(churned_customers)

# Section: Data Visualization
st.write("## Churn Insights Visualization")

# 1. Pie Chart for Churn Distribution
fig_pie = px.pie(df, names='Churn', title='Churn Distribution', hole=0.3, color_discrete_sequence=['#FF4500', '#32CD32'])
st.plotly_chart(fig_pie)
st.write("This pie chart shows the churn distribution based on various features. Notably, we have a 79% churn rate compared to the bar graph of plans vs churn, thid is because here the data is taken by combaining all the features when compared to Pricing & plans vs churn bar graph")

# 2. Stacked Bar Chart for Pricing and Plans vs Churn (showing percentages)
pricing_churn = df.groupby(['PricingAndPlans', 'Churn']).size().unstack().fillna(0)
pricing_churn_percentage = (pricing_churn.div(pricing_churn.sum(axis=1), axis=0) * 100).round(2)
fig_stacked_bar = px.bar(pricing_churn_percentage, x=pricing_churn_percentage.index, 
                          y=pricing_churn_percentage.columns, 
                          title='Churn Distribution by Pricing and Plans (Percentage)', 
                          labels={'value': 'Percentage', 'variable': 'Churn'},
                          text_auto=True,
                          color_discrete_sequence=['#FF4500', '#32CD32'])
st.plotly_chart(fig_stacked_bar)

# 3. Heatmap for Feature Correlation
st.write("### Feature Correlation Heatmap")
st.write("This heatmap displays the correlations between numeric features in the dataset.")
numeric_df = df.select_dtypes(include=[np.number])
correlation_matrix = numeric_df.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='viridis', fmt='.2f', center=0)
st.pyplot(plt)

# 4. Bar Plot for Customer Engagement vs Churn
customer_engagement_churn = df.groupby('Churn')['CustomerEngagement'].mean().reset_index()

# Create a color map manually
color_map = {0: '#FF0000', 1: '#00FF00'}  # Red for not churned, green for churned
customer_engagement_churn['Color'] = customer_engagement_churn['Churn'].map(color_map)

# Create the bar plot with the specified colors
fig_bar = px.bar(customer_engagement_churn, 
                  x='Churn', 
                  y='CustomerEngagement',
                  title='Average Customer Engagement by Churn Status', 
                  labels={'CustomerEngagement': 'Average Engagement'},
                  color='Color',  # Use the new 'Color' column for coloring
                  color_discrete_map=color_map)  # Use the color map directly
st.plotly_chart(fig_bar)


# 5. Histogram for Customer Sentiment
fig_histogram = px.histogram(df, x='CustomerSentiment', color='Churn', 
                              title='Distribution of Customer Sentiment by Churn Status',
                              barmode='group',
                              color_discrete_sequence=['#32CD32', '#FF4500'])
st.plotly_chart(fig_histogram)

# 6. Scatter Plot for Customer Engagement vs Usage Patterns
fig_scatter = px.scatter(df, x='CustomerEngagement', y='UsagePatterns',
                         color='Churn', title='Customer Engagement vs Usage Patterns',
                         color_continuous_scale=px.colors.sequential.Viridis,  
                         opacity=0.8,
                         range_color=[0, 1])  
st.plotly_chart(fig_scatter)

# Sidebar Layout
st.sidebar.write("## Churn Prediction")
# Display the prediction result only in the sidebar
if 'churn_probability' in st.session_state:
    st.sidebar.write(f"### Churn Probability: {st.session_state.churn_probability:.4f}")
    if st.session_state.churn_probability > 0.5:
        st.sidebar.warning("The customer is likely to churn.")
    else:
        st.sidebar.success("The customer is unlikely to churn.")

# Sidebar form for input
with st.sidebar.form("churn_prediction_form"):
    submit_button = st.form_submit_button(label="Predict Churn")

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

    # Handle categorical features encoding
    def encode_features(pricing_and_plans, contractual_obligations, customer_lifecycle_stage):
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
        st.session_state.churn_probability = churn_probability
