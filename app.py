import streamlit as st
import pandas as pd
import joblib

# Load the model, feature names, and LabelEncoder
model = joblib.load('student_performance_model.pkl')
feature_names = joblib.load('feature_names.pkl')
label_encoder = joblib.load('label_encoder.pkl')  # Load the saved LabelEncoder

# Streamlit app header
st.title("Student Performance Prediction")

# File uploader widget
uploaded_file = st.file_uploader("Upload Student Data CSV", type=["csv"])

if uploaded_file is not None:
    # Load uploaded file
    new_data = pd.read_csv(uploaded_file)

    # Ensure the dataset contains 'id_student' and required features
    if 'id_student' not in new_data.columns:
        st.error("The uploaded dataset must contain the 'id_student' column.")
    else:
        missing_features = [feature for feature in feature_names if feature not in new_data.columns]
        if missing_features:
            st.error(f"The uploaded dataset is missing the following required features: {missing_features}")
        else:
            # Select only the features used during training
            X = new_data[feature_names]

            # Handle missing values (same as training preprocessing)
            X.fillna({'score': 0, 'sum_click': 0}, inplace=True)

            # Predict student performance
            predictions = model.predict(X)

            # Decode predictions to original class labels
            decoded_predictions = label_encoder.inverse_transform(predictions)

            # Combine student IDs with predictions
            results = pd.DataFrame({
                'id_student': new_data['id_student'],
                'Prediction': decoded_predictions
            })

            # Display predictions in the app
            st.write("Predictions:", results)

            # Provide a download link for predictions
            st.download_button(
                label="Download Predictions",
                data=results.to_csv(index=False),
                file_name="predictions_with_student_ids.csv"
            )
