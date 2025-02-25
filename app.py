# # app.py

# import streamlit as st
# import pandas as pd
# import numpy as np
# import joblib
# import os

# from sklearn.tree import DecisionTreeClassifier
# from sklearn.neural_network import MLPClassifier
# from sklearn.ensemble import RandomForestClassifier

# from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
# import seaborn as sns
# import matplotlib.pyplot as plt

# # Set Streamlit page configuration
# st.set_page_config(
#     page_title="Lung Capacity Predictor",
#     page_icon="🌬️",
#     layout="centered",
#     initial_sidebar_state="expanded",
# )

# # Title and description
# st.title("🌬️ Lung Capacity Predictor Using Spirometry Data")
# st.markdown("""
# Welcome to the **Lung Capacity Predictor** app! Input your spirometry measurements below to predict your lung capacity category.
# """)

# # Sidebar for user inputs
# st.sidebar.header("User Input Parameters")

# def user_input_features():
#     FVC = st.sidebar.number_input("Forced Vital Capacity (FVC) [liters]", min_value=0.0, max_value=10.0, value=4.5, step=0.1)
#     FEV1 = st.sidebar.number_input("Forced Expiratory Volume in 1 Second (FEV1) [liters]", min_value=0.0, max_value=10.0, value=3.5, step=0.1)
#     # Add more input fields if necessary

#     data = {'FVC': FVC,
#             'FEC1': FEV1}
#     features = pd.DataFrame(data, index=[0])
#     return features

# input_df = user_input_features()

# # Display user input
# st.subheader("🔍 Your Input Parameters")
# st.write(input_df)

# # Load Models
# @st.cache_resource()
# def load_models():
#     decision_tree = joblib.load("decision_tree_model.pkl")
#     ann_model = joblib.load( "ann_model.pkl")
#     rf_model = joblib.load( "rf_model.pkl")
#     return decision_tree, ann_model, rf_model

# decision_tree, ann_model, rf_model = load_models()

# # Prediction Function
# def predict(models, input_data):
#     predictions = {}
#     for name, model in models.items():
#         pred = model.predict(input_data)[0]
#         predictions[name] = pred
#     return predictions

# # Mapping labels
# label_mapping = {0: 'Normal', 1: 'Obstructed', 2: 'Restricted', 3: 'Unclassified'}

# # Prepare models dictionary
# models = {
#     "Decision Tree": decision_tree,
#     "Artificial Neural Network (ANN)": ann_model,
#     "Random Forest": rf_model
# }

# # Predict button
# if st.button("🔮 Predict Lung Capacity"):
#     predictions = predict(models, input_df)
#     st.subheader("📊 Prediction Results")
    
#     # Display predictions
#     for model_name, pred in predictions.items():
#         st.write(f"**{model_name}:** {label_mapping.get(pred, 'Unknown')}")

#     # Optionally, display probability scores if models support it
#     # Example for models with predict_proba
#     st.subheader("📈 Prediction Probabilities")
#     prob_df = pd.DataFrame()
#     for model_name, model in models.items():
#         if hasattr(model, "predict_proba"):
#             proba = model.predict_proba(input_df)[0]
#             prob_df[model_name] = proba
#     if not prob_df.empty:
#         prob_df.index = ['Normal', 'Obstructed', 'Restricted', 'Unclassified']
#         st.write(prob_df)

#     # Visualizing the predictions (e.g., bar chart of probabilities)
#     if not prob_df.empty:
#         st.markdown("### 🌟 Prediction Probabilities by Model")
#         fig, ax = plt.subplots(figsize=(10, 6))
#         prob_df.plot(kind='bar', ax=ax)
#         plt.xlabel('Lung Capacity Category')
#         plt.ylabel('Probability')
#         plt.title('Prediction Probabilities by Model')
#         plt.legend(title='Models')
#         st.pyplot(fig)

# # Additional Styling and Information
# st.markdown("""
# ---
# ### 📚 About This App

# This application utilizes **Machine Learning** models to predict lung capacity categories based on spirometry measurements. The models included are:

# - **Decision Tree**
# - **Artificial Neural Network (ANN)**
# - **Random Forest**

# ### 🔧 How It Works

# 1. **Input**: Enter your FVC and FEV1 values.
# 2. **Predict**: Click the **Predict** button to see the lung capacity category.
# 3. **Results**: View predictions from different models and their probabilities.

# ### 🚀 Future Enhancements

# - Incorporate more features for improved accuracy.
# - Add more advanced visualization and analysis.
# - Deploy the app for broader accessibility.

# ### 🛠️ Developer

# Developed by [Your Name]. For more information, contact [your.email@example.com](mailto:your.email@example.com).

# """)
# app.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

# Set Streamlit page configuration
st.set_page_config(
    page_title="🌬️ Lung Capacity Predictor",
    page_icon="🌬️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Load Models
@st.cache_resource()
def load_models():
    decision_tree = joblib.load("decision_tree_model.pkl")
    ann_model = joblib.load("ann_model.pkl")
    rf_model = joblib.load("rf_model.pkl")
    return decision_tree, ann_model, rf_model

decision_tree, ann_model, rf_model = load_models()

# Mapping labels
label_mapping = {0: 'Normal', 1: 'Obstructed', 2: 'Restricted', 3: 'Unclassified'}

# Sidebar Content
st.sidebar.title("🌟 About the App")
st.sidebar.markdown("""
Welcome to the **Lung Capacity Predictor** app! This tool utilizes **Machine Learning** models to predict your lung capacity category based on your spirometry measurements.

### 🔍 **How to Use**
1. **Input Parameters**: Enter your spirometry measurements in the sidebar.
2. **Predict**: Click the **Predict Lung Capacity** button to generate predictions.
3. **Results**: View predictions from individual models and the hybrid model.

### 📈 **Models Used**
- **Decision Tree**
- **Artificial Neural Network (ANN)**
- **Random Forest**
- **Hybrid Model** (Mode of individual model predictions)

### 🛠️ **Developer**
Developed by [Your Name]. For inquiries, contact [your.email@example.com](mailto:your.email@example.com).
""")

# Title and description
st.title("🌬️ Applying Hybrid Machine Learning Algorithm to Measure Total Lung Capacity from Spirometric Values")
st.markdown("""
Enter your spirometry measurements below to predict your lung capacity category. The app utilizes multiple machine learning models to provide comprehensive predictions.
""")

# Sidebar for user inputs
st.sidebar.header("📝 Input Parameters")

def user_input_features():
    FVC = st.sidebar.number_input(
        "Forced Vital Capacity (FVC) [liters]",
        min_value=0.0,
        max_value=10.0,
        value=4.5,
        step=0.1,
        help="Total volume of air that can be forcefully exhaled after full inhalation."
    )
    FEV1 = st.sidebar.number_input(
        "Forced Expiratory Volume in 1 Second (FEV1) [liters]",
        min_value=0.0,
        max_value=10.0,
        value=3.5,
        step=0.1,
        help="Volume of air exhaled forcefully in one second."
    )
    # Add more input fields if necessary (e.g., AGE, smoking status)

    data = {'FVC': FVC,
            'FEC1': FEV1}
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# Display user input
st.subheader("🔍 Your Input Parameters")
st.write(input_df)

# Prediction Function
def predict(models, input_data):
    predictions = {}
    for name, model in models.items():
        pred = model.predict(input_data)[0]
        predictions[name] = pred
    return predictions



from scipy import stats

def hybrid_predict(models, input_data):
    preds = [model.predict(input_data)[0] for model in models.values()]
    p=np.bincount(preds).argmax()  # Ensure compatibility with current scipy version
    print(preds)
    return p

from scipy import stats

# Prepare models dictionary
models = {
    "Decision Tree": decision_tree,
    "Artificial Neural Network (ANN)": ann_model,
    "Random Forest": rf_model
}

# Predict button
if st.button("🔮 Predict Lung Capacity"):
    with st.spinner("Predicting..."):
        # Individual model predictions
        individual_predictions = predict(models, input_df)
        
        # Hybrid model prediction
        hybrid_pred = hybrid_predict(models, input_df)
        
        # Display individual predictions
        st.subheader("📊 Prediction Results from Individual Models")
        pred_df = pd.DataFrame.from_dict(individual_predictions, orient='index', columns=['Prediction'])
        pred_df['Prediction'] = pred_df['Prediction'].map(label_mapping)
        st.table(pred_df)
        
        # Display hybrid model prediction
        st.subheader("🎯 Hybrid Model Prediction")
        st.write(f"**Hybrid Model:** {label_mapping.get(hybrid_pred, 'Unknown')}")
        
        # Optional: Display probability scores if models support it
        st.subheader("📈 Prediction Probabilities")
        prob_df = pd.DataFrame()
        for model_name, model in models.items():
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(input_df)[0]
                prob_df[model_name] = proba
        if not prob_df.empty:
            prob_df.index = ['Normal', 'Obstructed', 'Restricted', 'Unclassified']
            st.write(prob_df)
        
            # Visualizing the predictions (e.g., bar chart of probabilities)
            st.markdown("### 🌟 Prediction Probabilities by Model")
            fig, ax = plt.subplots(figsize=(10, 6))
            prob_df.plot(kind='bar', ax=ax)
            plt.xlabel('Lung Capacity Category')
            plt.ylabel('Probability')
            plt.title('Prediction Probabilities by Model')
            plt.legend(title='Models')
            st.pyplot(fig)
        
        # Visualization: Confusion Matrix (Optional, if you want to display on the app)
        # This would require test data; if not available, skip or provide based on your context.

# # Additional Styling and Information
# st.markdown("""
# ---
# ### 📚 About This App

# This application utilizes **Machine Learning** models to predict lung capacity categories based on spirometry measurements. The models included are:

# - **Decision Tree**
# - **Artificial Neural Network (ANN)**
# - **Random Forest**
# - **Hybrid Model** (Mode of individual model predictions)

# ### 🔧 How It Works

# 1. **Input**: Enter your FVC and FEV1 values in the sidebar.
# 2. **Predict**: Click the **Predict Lung Capacity** button to generate predictions.
# 3. **Results**: View predictions from individual models and the hybrid model.

# ### 🚀 Future Enhancements

# - Incorporate more features for improved accuracy (e.g., AGE, Smoking Status).
# - Add more advanced visualizations and analyses.
# - Deploy the app for broader accessibility.

# ### 🛠️ Developer

# Developed by [Your Name]. For more information, contact [your.email@example.com](mailto:your.email@example.com).
# """)
