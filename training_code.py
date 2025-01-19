import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib

# Load datasets
assessments = pd.read_csv('assessments.csv')
courses = pd.read_csv('courses.csv')
student_assessment = pd.read_csv('studentAssessment.csv', nrows=1000)  # Limit rows to save memory
student_info = pd.read_csv('studentInfo.csv', nrows=1000)
student_registration = pd.read_csv('studentRegistration.csv', nrows=1000)
student_vle = pd.read_csv('studentVle.csv', nrows=1000)
vle = pd.read_csv('vle.csv')

# Merge datasets to form master_data
vle_data = pd.merge(student_vle, vle, on=['id_site', 'code_module', 'code_presentation'], how='left')
assessment_data = pd.merge(student_assessment, assessments, on='id_assessment', how='left')
assessment_data = pd.merge(assessment_data, student_info, on=['id_student', 'code_module', 'code_presentation'], how='left')
master_data = pd.merge(assessment_data, vle_data, on=['id_student', 'code_module', 'code_presentation'], how='left')

# Handle missing values
master_data.fillna({'score': 0, 'sum_click': 0}, inplace=True)

# Feature engineering
if 'sum_click' in master_data.columns:
    master_data['total_clicks'] = master_data.groupby('id_student')['sum_click'].transform('sum')
else:
    print("Column 'sum_click' is missing. Skipping 'total_clicks' feature.")

# Select relevant columns
relevant_columns = ['id_student', 'score', 'sum_click', 'total_clicks', 'final_result']
master_data = master_data[relevant_columns]

# Encode categorical features
label_encoder = LabelEncoder()
master_data['final_result'] = label_encoder.fit_transform(master_data['final_result'])

# Prepare data for training
X = master_data.drop(columns=['final_result', 'id_student'])
y = master_data['final_result']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier(n_estimators=50, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Save the model, feature names, and LabelEncoder
joblib.dump(model, 'student_performance_model.pkl')
joblib.dump(X_train.columns.tolist(), 'feature_names.pkl')
joblib.dump(label_encoder, 'label_encoder.pkl')
print("Model, feature names, and LabelEncoder saved.")
