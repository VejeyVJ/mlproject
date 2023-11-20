from flask import Flask, render_template, request
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

app = Flask(__name__)

# Load the dataset
data = pd.read_csv("C:/Users/jelly/Downloads/archive (2)/indian_liver_patient.csv")


# Handle missing values (if any)
data.dropna(inplace=True)

# Separate features (X) and target variable (y)
X = data.drop('Dataset', axis=1)
y = data['Dataset']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Separate numerical and categorical features
numeric_features = X.select_dtypes(include=['float64']).columns
categorical_features = X.select_dtypes(include=['object']).columns

# Create transformers for numerical and categorical features
numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combine transformers into a preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Create a RandomForestClassifier model
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

# Train the model
model.fit(X_train, y_train)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    features = {
        'Age': float(request.form['Age']),
        'Total_Bilirubin': float(request.form['Total_Bilirubin']),
        'Direct_Bilirubin': float(request.form['Direct_Bilirubin']),
        'Alkaline_Phosphotase': float(request.form['Alkaline_Phosphotase']),
        'Alamine_Aminotransferase': float(request.form['Alamine_Aminotransferase']),
        'Aspartate_Aminotransferase': float(request.form['Aspartate_Aminotransferase']),
        'Total_Protiens': float(request.form['Total_Protiens']),
        'Albumin': float(request.form['Albumin']),
        'Albumin_and_Globulin_Ratio': float(request.form['Albumin_and_Globulin_Ratio']),
        'Gender': request.form['Gender']
    }

    input_data = pd.DataFrame([features])
    prediction = model.predict(input_data)

    result = "Liver Disease Detected" if prediction[0] == 1 else "No Liver Disease Detected"
    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
