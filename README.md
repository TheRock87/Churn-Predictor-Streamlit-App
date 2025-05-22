# Customer Churn Predictor

A Streamlit web application that predicts customer churn probability using machine learning.

## Features

- Interactive web interface for entering customer data
- Real-time churn probability prediction
- Data visualizations to understand churn patterns
- Modern, responsive UI
- Sample data option for quick testing

## Setup and Installation

1. Clone this repository
2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```
3. Place your trained model file (pickle format) in the root directory as `model.pkl`
4. Run the Streamlit app:
   ```
   streamlit run app.py
   ```

## Deployment

This app can be deployed to Streamlit Cloud for free:

1. Push this code to a GitHub repository
2. Sign up for [Streamlit Cloud](https://streamlit.io/cloud)
3. Connect your GitHub repository
4. Deploy with a few clicks

## Model Information

The model expects the following features:
- gender
- SeniorCitizen
- Partner
- Dependents
- tenure
- PhoneService
- MultipleLines
- InternetService
- OnlineSecurity
- OnlineBackup
- DeviceProtection
- TechSupport
- StreamingTV
- StreamingMovies
- Contract
- PaperlessBilling
- PaymentMethod
- MonthlyCharges
- TotalCharges

## License

MIT
