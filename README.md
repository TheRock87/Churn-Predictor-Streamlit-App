# Customer Churn Predictor

![Churn Prediction](https://img.shields.io/badge/ML-Churn%20Prediction-blue) ![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B) ![Python](https://img.shields.io/badge/Python-3.7+-yellow)

**Live Demo:** [Churn Predictor App](https://churn-predictor-app-fbh.streamlit.app)

A machine learning application that predicts customer churn for a telecommunications company with an interactive user interface. This tool helps businesses identify customers at risk of leaving, enabling proactive retention strategies.

## Table of Contents
- [Demo](#demo)
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Usage](#usage)
- [Machine Learning Model](#machine-learning-model)
- [Business Impact](#business-impact)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [Contact](#contact)

## Demo

![Demo GIF](https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExNzMzZDIyMTBkNmQ4ZDcxZDZmZGQ0ZTU1MTkxZTI5OGE0ZjVmNzEzZiZlcD12MV9pbnRlcm5hbF9naWZzX2dpZklkJmN0PWc/your-demo-gif-id/giphy.gif)

Try the live demo: [Churn Predictor App](https://churn-predictor-app-fbh.streamlit.app)

## Features

- **Real-time Churn Prediction**: Input customer data and get immediate churn probability
- **Interactive UI**: User-friendly interface for easy data input and visualization
- **Comprehensive Analysis**: View detailed breakdown of prediction results
- **Responsive Design**: Works on desktop and mobile devices

## Technologies Used

- **Frontend**: Streamlit
- **Machine Learning**: 
  - XGBoost
  - Random Forest
  - Scikit-learn
- **Data Processing**: 
  - Pandas
  - NumPy
- **Visualization**: 
  - Plotly
  - Matplotlib
  - Seaborn

## Installation

To run this project locally:

```bash
# Clone the repository
git clone [https://github.com/yourusername/customer-churn-predictor.git](https://github.com/yourusername/customer-churn-predictor.git)
cd customer-churn-predictor

# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```
Usage
-----

1.  streamlit run app.py or visit the [live demo](https://churn-predictor-app-fbh.streamlit.app/)
    
2.  Fill in the customer information form with relevant data
    
3.  Click "Predict Churn Risk" to get the prediction result
    
4.  View the prediction probability and contributing factors
    
5.  Use the insights to develop targeted retention strategies
    

Machine Learning Model
----------------------

### Model Architecture

The project uses an ensemble model that combines:

*   **Random Forest Classifier** (with 200 estimators)
    
*   **XGBoost Classifier**
    

The ensemble employs a soft voting approach, averaging predicted probabilities from both models to make the final prediction. This approach provides more robust predictions than either model alone.

### Data Preprocessing

*   **Handling Imbalanced Data**: SMOTE (Synthetic Minority Over-sampling Technique) is used to address class imbalance in the training data
    
*   **Feature Scaling**: Numerical features are standardized using StandardScaler
    
*   **Categorical Encoding**: Categorical variables are appropriately encoded
    

### Model Performance

The ensemble model achieves:

*   **Accuracy**: 75.7%
    
*   **ROC-AUC Score**: 74.7%
    
*   **Precision for Churn Class**: 53%
    
*   **Recall for Churn Class**: 72%
    
*   **F1-Score for Churn Class**: 61%
    

A prediction threshold of 0.415 was selected to optimize the balance between precision and recall, prioritizing the identification of customers at risk of churning.

### Key Features

The model considers various customer attributes, with the most influential being:

*   Contract type (month-to-month contracts have higher churn risk)
    
*   Tenure (newer customers are more likely to churn)
    
*   Monthly charges (higher charges correlate with increased churn)
    
*   Payment method
    
*   Internet service type
    

Business Impact
---------------

This churn prediction tool provides significant business value:

*   **Reduced Revenue Loss**: Identify at-risk customers before they leave
    
*   **Targeted Retention**: Focus retention efforts on customers most likely to churn
    
*   **Cost Efficiency**: Optimize marketing and retention budgets
    
*   **Customer Insights**: Understand key factors driving customer churn
    
*   **Improved Customer Experience**: Address issues before they lead to customer departure
    

By implementing proactive retention strategies based on the model's predictions, businesses can expect to reduce churn rates by 10-25%, resulting in substantial revenue preservation.

Roadmap
-------

Future enhancements planned for this project:

*   \[ \] Add feature importance visualization for individual predictions
    
*   \[ \] Implement A/B testing framework for retention strategies
    
*   \[ \] Develop API endpoints for integration with CRM systems
    
*   \[ \] Add time-series analysis for churn prediction over time
    
*   \[ \] Create customer segmentation based on churn risk profiles
    

Contributing
------------

Contributions are welcome! Please feel free to submit a Pull Request.

1.  Fork the repository
    
2.  git checkout -b feature/AmazingFeature)
    
3.  git commit -m 'Add some AmazingFeature')
    
4.  git push origin feature/AmazingFeature)
    
5.  Open a Pull Request
    

Contact
-------

Your Name - [hossam.kharbotly@gmail.com](mailto:hossam.kharbotly@gmail.com)

Project Link: [https://github.com/TheRock87/Churn-Predictor-Streamlit-App](https://github.com/TheRock87/Churn-Predictor-Streamlit-App)
