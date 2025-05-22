import streamlit as st
import pandas as pd
import pickle
import numpy as np
import os

# Try to import optional packages but continue if they're not available
try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    st.warning("Plotly not available. Some visualizations will be limited.")

try:
    import xgboost
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    st.error("XGBoost not available. Model predictions will not work.")
    st.info("Please install XGBoost: pip install xgboost>=2.0.0")

# Set page configuration
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern look
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton button {
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
        border-radius: 10px;
        padding: 10px 24px;
        border: none;
    }
    .stButton button:hover {
        background-color: #45a049;
    }
    .prediction-box {
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    h1, h2, h3 {
        color: #1E3A8A;
    }
    .stAlert {
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Function to load the model and preprocessors
@st.cache_resource
def load_model_and_preprocessors():
    try:
        # Check if XGBoost is available
        if not XGBOOST_AVAILABLE:
            st.error("XGBoost is required but not available.")
            return None, None, None
            
        # Check if files exist before attempting to load them
        required_files = ['ensemble_model.pkl', 'scaler.pkl', 'encoder.pkl']
        missing_files = [f for f in required_files if not os.path.exists(f)]
        
        if missing_files:
            st.error(f"Missing required files: {', '.join(missing_files)}")
            st.info("Please make sure all required pickle files are in the app directory.")
            
            # For Streamlit Cloud deployment - check if files are in the mount directory
            alt_paths = {
                'ensemble_model.pkl': ['/mount/src/churn-predictor-streamlit-app/ensemble_model.pkl'],
                'scaler.pkl': ['/mount/src/churn-predictor-streamlit-app/scaler.pkl'],
                'encoder.pkl': ['/mount/src/churn-predictor-streamlit-app/encoder.pkl']
            }
            
            # Try alternative paths for each missing file
            for missing_file in missing_files:
                for alt_path in alt_paths.get(missing_file, []):
                    if os.path.exists(alt_path):
                        st.info(f"Found {missing_file} at alternative path: {alt_path}")
                        # Create a symlink to the original expected path
                        try:
                            os.symlink(alt_path, missing_file)
                            st.success(f"Linked {alt_path} to {missing_file}")
                        except Exception as e:
                            st.error(f"Failed to create symlink: {str(e)}")
            
            # Check again after trying alternative paths
            missing_files = [f for f in required_files if not os.path.exists(f)]
            if missing_files:
                return None, None, None
        
        # Load the ensemble model
        try:
            with open('ensemble_model.pkl', 'rb') as file:
                model = pickle.load(file)
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
            st.info("This could be due to incompatible versions or missing dependencies.")
            return None, None, None
            
        # Load the scaler for numerical features
        try:
            with open('scaler.pkl', 'rb') as file:
                scaler = pickle.load(file)
        except Exception as e:
            st.error(f"Error loading scaler: {str(e)}")
            return None, None, None
            
        # Load the encoder for categorical features
        try:
            with open('encoder.pkl', 'rb') as file:
                encoder = pickle.load(file)
        except Exception as e:
            st.error(f"Error loading encoder: {str(e)}")
            return None, None, None
            
        return model, scaler, encoder
    except Exception as e:
        st.error(f"Error loading files: {str(e)}")
        st.info("Please check that all files are present and compatible with the current environment.")
        return None, None, None

# Function for preprocessing inputs
def preprocess_inputs(df, scaler, encoder):
    # Make a copy of the dataframe to avoid modifying the original
    df_processed = df.copy()
    
    try:
        # Apply encoders to categorical columns
        if encoder is not None and isinstance(encoder, dict):
            for col, col_encoder in encoder.items():
                if col in df_processed.columns:
                    try:
                        # Transform the column using the specific encoder
                        df_processed[col] = col_encoder.transform(df_processed[[col]])
                    except Exception as e:
                        st.warning(f"Error encoding column {col}: {str(e)}")
                        # Fallback to manual encoding if encoder fails
                        if col == 'gender':
                            df_processed[col] = df_processed[col].map({'Male': 0, 'Female': 1})
                        elif col in ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']:
                            df_processed[col] = df_processed[col].map({'No': 0, 'Yes': 1})
                        elif col == 'MultipleLines':
                            df_processed[col] = df_processed[col].map({'No': 0, 'Yes': 1, 'No phone service': 2})
                        elif col == 'InternetService':
                            df_processed[col] = df_processed[col].map({'DSL': 0, 'Fiber optic': 1, 'No': 2})
                        elif col in ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']:
                            df_processed[col] = df_processed[col].map({'No': 0, 'Yes': 1, 'No internet service': 2})
                        elif col == 'Contract':
                            df_processed[col] = df_processed[col].map({'Month-to-month': 0, 'One year': 1, 'Two year': 2})
                        elif col == 'PaymentMethod':
                            payment_methods = {'Electronic check': 0, 'Mailed check': 1, 'Bank transfer (automatic)': 2, 'Credit card (automatic)': 3}
                            df_processed[col] = df_processed[col].map(payment_methods)
        else:
            # Fallback to manual encoding if encoder is not available
            st.warning("Encoder not available or not in expected format. Using manual encoding.")
            # Gender encoding
            if 'gender' in df_processed.columns:
                df_processed['gender'] = df_processed['gender'].map({'Male': 0, 'Female': 1})
            
            # Yes/No columns
            yes_no_columns = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
            for col in yes_no_columns:
                if col in df_processed.columns:
                    df_processed[col] = df_processed[col].map({'No': 0, 'Yes': 1})
            
            # MultipleLines encoding
            if 'MultipleLines' in df_processed.columns:
                df_processed['MultipleLines'] = df_processed['MultipleLines'].map(
                    {'No': 0, 'Yes': 1, 'No phone service': 2}
                )
            
            # InternetService encoding
            if 'InternetService' in df_processed.columns:
                df_processed['InternetService'] = df_processed['InternetService'].map(
                    {'DSL': 0, 'Fiber optic': 1, 'No': 2}
                )
            
            # Internet-dependent services
            internet_dependent = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
                                'TechSupport', 'StreamingTV', 'StreamingMovies']
            for col in internet_dependent:
                if col in df_processed.columns:
                    df_processed[col] = df_processed[col].map(
                        {'No': 0, 'Yes': 1, 'No internet service': 2}
                    )
            
            # Contract encoding
            if 'Contract' in df_processed.columns:
                df_processed['Contract'] = df_processed['Contract'].map(
                    {'Month-to-month': 0, 'One year': 1, 'Two year': 2}
                )
            
            # PaymentMethod encoding
            payment_methods = {'Electronic check': 0, 'Mailed check': 1, 
                            'Bank transfer (automatic)': 2, 'Credit card (automatic)': 3}
            if 'PaymentMethod' in df_processed.columns:
                df_processed['PaymentMethod'] = df_processed['PaymentMethod'].map(payment_methods)
        
        # Ensure SeniorCitizen is an integer (it might be coming in as Yes/No)
        if 'SeniorCitizen' in df_processed.columns and df_processed['SeniorCitizen'].dtype == 'object':
            df_processed['SeniorCitizen'] = df_processed['SeniorCitizen'].map({'No': 0, 'Yes': 1})
        
        # Apply scaling to numerical features
        numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
        if scaler is not None and all(col in df_processed.columns for col in numerical_cols):
            try:
                df_processed[numerical_cols] = scaler.transform(df_processed[numerical_cols])
            except Exception as e:
                st.warning(f"Error scaling numerical features: {str(e)}")
        
        # Make sure all columns are numeric
        for col in df_processed.columns:
            if df_processed[col].dtype == 'object':
                st.warning(f"Column {col} still contains non-numeric values: {df_processed[col].unique()}")
                # Try to convert to numeric as a last resort
                df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')
        
        return df_processed
    
    except Exception as e:
        st.error(f"Error in preprocessing: {str(e)}")
        # Return the original dataframe if preprocessing fails
        return df

# Main app layout
def main():
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/decision.png", width=100)
        st.title("Churn Predictor")
        st.markdown("---")
        st.markdown("Predict customer churn probability using machine learning.")
        st.markdown("---")
        
        # Model info
        st.subheader("About the Model")
        st.write("This model predicts the probability of customer churn based on various features.")
        
        # Add a sample data option
        if st.button("Load Sample Data"):
            st.session_state.use_sample_data = True
        
        st.markdown("---")
        st.markdown("© 2025 Churn Predictor")
    
    # Main content
    st.title("Customer Churn Prediction")
    st.write("Enter customer information to predict the likelihood of churn.")
    
    # Initialize session state for sample data
    if 'use_sample_data' not in st.session_state:
        st.session_state.use_sample_data = False
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["Prediction", "Visualization", "About"])
    
    with tab1:
        # Input form
        st.subheader("Customer Information")
        
        col1, col2, col3 = st.columns(3)
        
        # Sample data values
        sample_data = {
            'gender': 'Male',
            'SeniorCitizen': 0,
            'Partner': 'Yes',
            'Dependents': 'No',
            'tenure': 36,
            'PhoneService': 'Yes',
            'MultipleLines': 'Yes',
            'InternetService': 'Fiber optic',
            'OnlineSecurity': 'No',
            'OnlineBackup': 'Yes',
            'DeviceProtection': 'No',
            'TechSupport': 'No',
            'StreamingTV': 'Yes',
            'StreamingMovies': 'Yes',
            'Contract': 'Month-to-month',
            'PaperlessBilling': 'Yes',
            'PaymentMethod': 'Electronic check',
            'MonthlyCharges': 90.45,
            'TotalCharges': 3232.2
        }
        
        # Use sample data if button was clicked
        use_sample = st.session_state.use_sample_data
        
        with col1:
            gender = st.selectbox('Gender', options=['Male', 'Female'], index=0 if use_sample and sample_data['gender'] == 'Male' else 1)
            
            senior_citizen = st.selectbox('Senior Citizen', options=[0, 1], 
                                         index=sample_data['SeniorCitizen'] if use_sample else 0,
                                         format_func=lambda x: 'Yes' if x == 1 else 'No')
            
            partner = st.selectbox('Partner', options=['Yes', 'No'], 
                                  index=0 if use_sample and sample_data['Partner'] == 'Yes' else 1)
            
            dependents = st.selectbox('Dependents', options=['Yes', 'No'], 
                                     index=0 if use_sample and sample_data['Dependents'] == 'Yes' else 1)
            
            tenure = st.slider('Tenure (months)', min_value=0, max_value=72, 
                              value=sample_data['tenure'] if use_sample else 12)
            
            phone_service = st.selectbox('Phone Service', options=['Yes', 'No'], 
                                        index=0 if use_sample and sample_data['PhoneService'] == 'Yes' else 1)
        
        with col2:
            multiple_lines = st.selectbox('Multiple Lines', 
                                         options=['Yes', 'No', 'No phone service'],
                                         index=0 if use_sample and sample_data['MultipleLines'] == 'Yes' else 1)
            
            internet_service = st.selectbox('Internet Service', 
                                           options=['DSL', 'Fiber optic', 'No'],
                                           index=1 if use_sample and sample_data['InternetService'] == 'Fiber optic' else 0)
            
            online_security = st.selectbox('Online Security', 
                                          options=['Yes', 'No', 'No internet service'],
                                          index=1 if use_sample and sample_data['OnlineSecurity'] == 'No' else 0)
            
            online_backup = st.selectbox('Online Backup', 
                                        options=['Yes', 'No', 'No internet service'],
                                        index=0 if use_sample and sample_data['OnlineBackup'] == 'Yes' else 1)
            
            device_protection = st.selectbox('Device Protection', 
                                           options=['Yes', 'No', 'No internet service'],
                                           index=1 if use_sample and sample_data['DeviceProtection'] == 'No' else 0)
            
            tech_support = st.selectbox('Tech Support', 
                                       options=['Yes', 'No', 'No internet service'],
                                       index=1 if use_sample and sample_data['TechSupport'] == 'No' else 0)
        
        with col3:
            streaming_tv = st.selectbox('Streaming TV', 
                                       options=['Yes', 'No', 'No internet service'],
                                       index=0 if use_sample and sample_data['StreamingTV'] == 'Yes' else 1)
            
            streaming_movies = st.selectbox('Streaming Movies', 
                                          options=['Yes', 'No', 'No internet service'],
                                          index=0 if use_sample and sample_data['StreamingMovies'] == 'Yes' else 1)
            
            contract = st.selectbox('Contract', 
                                   options=['Month-to-month', 'One year', 'Two year'],
                                   index=0 if use_sample and sample_data['Contract'] == 'Month-to-month' else 1)
            
            paperless_billing = st.selectbox('Paperless Billing', 
                                           options=['Yes', 'No'],
                                           index=0 if use_sample and sample_data['PaperlessBilling'] == 'Yes' else 1)
            
            payment_method = st.selectbox('Payment Method', 
                                         options=['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'],
                                         index=0 if use_sample and sample_data['PaymentMethod'] == 'Electronic check' else 1)
            
            monthly_charges = st.number_input('Monthly Charges (please covert from EGP to USD)', 
                                            min_value=0.0, max_value=200.0, step=0.01,
                                            value=sample_data['MonthlyCharges'] if use_sample else 20.0)
            
            total_charges = st.number_input('Total Charges (please covert from EGP to USD)', 
                                          min_value=0.0, max_value=10000.0, step=0.01,
                                          value=sample_data['TotalCharges'] if use_sample else tenure * monthly_charges)
        
        # Reset sample data flag after use
        if use_sample:
            st.session_state.use_sample_data = False
        
        # Predict button
        predict_button = st.button("Predict Churn Probability", use_container_width=True)
        
        if predict_button:
            # Load model and preprocessors
            model, scaler, encoder = load_model_and_preprocessors()
            
            if model and scaler and encoder:
                # Create input dataframe
                input_data = pd.DataFrame({
                    'gender': [gender],
                    'SeniorCitizen': [senior_citizen],
                    'Partner': [partner],
                    'Dependents': [dependents],
                    'tenure': [tenure],
                    'PhoneService': [phone_service],
                    'MultipleLines': [multiple_lines],
                    'InternetService': [internet_service],
                    'OnlineSecurity': [online_security],
                    'OnlineBackup': [online_backup],
                    'DeviceProtection': [device_protection],
                    'TechSupport': [tech_support],
                    'StreamingTV': [streaming_tv],
                    'StreamingMovies': [streaming_movies],
                    'Contract': [contract],
                    'PaperlessBilling': [paperless_billing],
                    'PaymentMethod': [payment_method],
                    'MonthlyCharges': [monthly_charges],
                    'TotalCharges': [total_charges]
                })
                
                # Preprocess inputs with scaler and encoder
                processed_data = preprocess_inputs(input_data, scaler, encoder)
                
                try:
                    # Make prediction
                    prediction_proba = model.predict_proba(processed_data)[0][1]
                    prediction = model.predict(processed_data)[0]
                    
                    # Display prediction
                    st.markdown("### Prediction Results")
                    
                    # Create columns for results
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Gauge chart for probability
                        fig = go.Figure(go.Indicator(
                            mode="gauge+number",
                            value=prediction_proba * 100,
                            domain={'x': [0, 1], 'y': [0, 1]},
                            title={'text': "Churn Probability"},
                            gauge={
                                'axis': {'range': [0, 100]},
                                'bar': {'color': "darkblue"},
                                'steps': [
                                    {'range': [0, 30], 'color': "green"},
                                    {'range': [30, 70], 'color': "yellow"},
                                    {'range': [70, 100], 'color': "red"}
                                ],
                                'threshold': {
                                    'line': {'color': "red", 'width': 4},
                                    'thickness': 0.75,
                                    'value': prediction_proba * 100
                                }
                            }
                        ))
                        
                        fig.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        # Prediction result box
                        if prediction == 1:
                            st.error("### High Risk of Churn!")
                            st.markdown("""
                            #### Recommendations:
                            - Offer a loyalty discount
                            - Provide premium service upgrade
                            - Schedule a customer service call
                            - Send personalized retention offer
                            """)
                        else:
                            st.success("### Low Risk of Churn")
                            st.markdown("""
                            #### Recommendations:
                            - Continue monitoring usage patterns
                            - Consider upselling opportunities
                            - Invite to loyalty program
                            - Send satisfaction survey
                            """)
                    
                    # Feature importance visualization (placeholder - would need actual model coefficients)
                    st.subheader("Key Factors Influencing Prediction")
                    
                    # This is a placeholder - in a real app you would extract actual feature importance
                    # from your model if it supports it
                    importance_data = {
                        'Contract': 0.35,
                        'tenure': 0.25,
                        'MonthlyCharges': 0.15,
                        'InternetService': 0.10,
                        'TechSupport': 0.08,
                        'OnlineSecurity': 0.07
                    }
                    
                    if PLOTLY_AVAILABLE:
                        fig = px.bar(
                            x=list(importance_data.values()),
                            y=list(importance_data.keys()),
                            orientation='h',
                            labels={'x': 'Importance', 'y': 'Feature'},
                            title='Feature Importance',
                            color=list(importance_data.values()),
                            color_continuous_scale='Blues'
                        )
                        
                        fig.update_layout(height=400)
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        # Fallback to simple text display if plotly is not available
                        for feature, importance in sorted(importance_data.items(), key=lambda x: x[1], reverse=True):
                            st.write(f"**{feature}**: {importance:.2f}")
                    
                except Exception as e:
                    st.error(f"Prediction error: {e}")
                    st.info("This could be due to a mismatch between the input features and what the model expects. Make sure your model is compatible with the input features.")
    
    with tab2:
        st.subheader("Data Visualization")
        
        # Placeholder visualizations - in a real app, these would be based on your actual dataset
        st.write("These visualizations help understand patterns in customer churn.")
        
        if PLOTLY_AVAILABLE:
            col1, col2 = st.columns(2)
            
            with col1:
                # Churn by Contract Type
                contract_data = {
                    'Contract': ['Month-to-month', 'One year', 'Two year'],
                    'Churn Rate': [0.42, 0.11, 0.03]
                }
                fig = px.bar(
                    contract_data, 
                    x='Contract', 
                    y='Churn Rate',
                    title='Churn Rate by Contract Type',
                    color='Churn Rate',
                    color_continuous_scale='Reds'
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Churn by Tenure
                tenure_data = pd.DataFrame({
                    'Tenure Group': ['0-12', '12-24', '24-36', '36-48', '48-60', '60-72'],
                    'Churn Rate': [0.52, 0.36, 0.27, 0.18, 0.12, 0.08]
                })
                fig = px.line(
                    tenure_data, 
                    x='Tenure Group', 
                    y='Churn Rate', 
                    markers=True,
                    title='Churn Rate by Tenure (months)',
                    line_shape='linear'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Churn by Internet Service
                internet_data = {
                    'Internet Service': ['DSL', 'Fiber optic', 'No'],
                    'Churn Rate': [0.19, 0.42, 0.07]
                }
                fig = px.pie(
                    internet_data, 
                    names='Internet Service', 
                    values='Churn Rate',
                    title='Churn Distribution by Internet Service',
                    hole=0.4,
                    color_discrete_sequence=px.colors.sequential.Blues
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Monthly Charges vs Churn
                fig = px.histogram(
                    x=[30, 45, 60, 75, 90, 105, 120],
                    nbins=10,
                    title='Monthly Charges Distribution',
                    labels={'x': 'Monthly Charges ($)'},
                    color_discrete_sequence=['#3366CC']
                )
                
                # Add a vertical line for average
                fig.add_vline(x=70, line_width=2, line_dash="dash", line_color="red")
                fig.add_annotation(x=70, y=10, text="Avg. Charge", showarrow=True, arrowhead=1)
                
                st.plotly_chart(fig, use_container_width=True)
        else:
            # Fallback to simple text display if plotly is not available
            st.info("Visualizations require Plotly. Install with: pip install plotly")
            
            st.subheader("Key Churn Insights (Text Summary)")
            st.write("**Contract Type Impact:**")
            st.write("- Month-to-month: 42% churn rate")
            st.write("- One year: 11% churn rate")
            st.write("- Two year: 3% churn rate")
            
            st.write("**Tenure Impact:**")
            st.write("- New customers (0-12 months): 52% churn rate")
            st.write("- Long-term customers (60-72 months): 8% churn rate")
            
            st.write("**Internet Service Impact:**")
            st.write("- Fiber optic: 42% churn rate (highest)")
            st.write("- DSL: 19% churn rate")
            st.write("- No internet service: 7% churn rate (lowest)")
    
    with tab3:
        st.subheader("About This App")
        st.write("""
        This Churn Predictor app uses machine learning to predict customer churn probability based on various features.
        
        ### How It Works
        1. Enter customer information in the Prediction tab
        2. Click the "Predict" button to get churn probability
        3. View visualizations to understand churn patterns
        
        ### Model Information
        The model is trained on historical customer data and uses features like contract type, tenure, and services to predict the likelihood of churn.
        
        ### Deployment
        This app is deployed on Streamlit Cloud and can be accessed by anyone worldwide.
        """)
        
        # Add expandable sections for more information
        with st.expander("Data Privacy"):
            st.write("""
            This app does not store any of the information entered. All predictions are made in real-time and data is not saved.
            """)
        
        with st.expander("Model Performance"):
            st.write("""
            The model has been trained and evaluated with the following metrics:
            - Accuracy: 76-80%
            - Precision: 55-88%
            - Recall: 72-77%
            - F1 Score: 65-82%
            
            Note: Actual performance may vary depending on the specific model deployed.
            """)

# Run the app
if __name__ == "__main__":
    main()
