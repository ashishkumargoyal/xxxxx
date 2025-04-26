import streamlit as st
import pandas as pd
import numpy as np
import joblib
import xgboost
from sklearn.preprocessing import LabelEncoder

# Set page configuration
st.set_page_config(page_title="Titanic Survival Prediction", page_icon="🚢", layout="centered")

# Title and description
st.title("🚢 Titanic Survival Prediction")
st.markdown("""
This app predicts whether a passenger would have survived the Titanic disaster based on their characteristics.
Enter the passenger details below and click **Predict** to see the result.
""")

# Load the pre-trained XGBoost model
try:
    model = joblib.load('xg_boost_model.pkl')
except FileNotFoundError:
    st.error("Model file 'xg_boost_model.pkl' not found. Please ensure it is in the same directory as app.py.")
    st.stop()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Create input form
with st.form("passenger_form"):
    st.header("Passenger Details")

    # Input fields for features
    pclass = st.selectbox("Passenger Class (Pclass)", options=[1, 2, 3], help="1 = 1st class, 2 = 2nd class, 3 = 3rd class")
    sex = st.selectbox("Sex", options=["male", "female"])
    age = st.slider("Age", min_value=0, max_value=100, value=30, step=1)
    sibsp = st.number_input("Number of Siblings/Spouses Aboard (SibSp)", min_value=0, max_value=8, value=0, step=1)
    parch = st.number_input("Number of Parents/Children Aboard (Parch)", min_value=0, max_value=6, value=0, step=1)
    fare = st.number_input("Fare", min_value=0.0, max_value=512.0, value=32.0, step=0.1)
    embarked = st.selectbox("Port of Embarkation (Embarked)", options=["C", "Q", "S"], help="C = Cherbourg, Q = Queenstown, S = Southampton")

    # Submit button
    submitted = st.form_submit_button("Predict")

# Process inputs and make prediction
if submitted:
    try:
        # Create a dictionary with input data
        input_data = {
            "Pclass": pclass,
            "Sex": sex,
            "Age": age,
            "SibSp": sibsp,
            "Parch": parch,
            "Fare": fare,
            "Embarked": embarked
        }

        # Convert to DataFrame
        input_df = pd.DataFrame([input_data])

        # Encode categorical variables (same as in the notebook)
        le_sex = LabelEncoder()
        input_df['Sex'] = le_sex.fit_transform(input_df['Sex'])

        le_embarked = LabelEncoder()
        input_df['Embarked'] = le_embarked.fit_transform(input_df['Embarked'])

        # Ensure the order of features matches the training data
        feature_order = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
        input_df = input_df[feature_order]

        # Make prediction
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0]

        # Display results
        st.subheader("Prediction Result")
        if prediction == 1:
            st.success("The passenger is predicted to **SURVIVE**! 🎉")
        else:
            st.error("The passenger is predicted to **NOT SURVIVE**. 😔")

        st.write(f"**Probability of Survival**: {probability[1]:.2%}")
        st.write(f"**Probability of Non-Survival**: {probability[0]:.2%}")

    except Exception as e:
        st.error(f"Error during prediction: {e}")

# Footer
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit | Model trained on Titanic dataset with XGBoost")