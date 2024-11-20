from flask import Flask, request, jsonify, render_template
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os

app = Flask(__name__)

# Load dataset and preprocess
df = pd.read_csv('Expanded_Destinations.csv')
expected_columns = ['Name', 'State', 'Type', 'BestTimeToVisit', 'Popularity']

if not all(col in df.columns for col in expected_columns):
    missing_columns = [col for col in expected_columns if col not in df.columns]
    raise KeyError(f"The following columns are missing from your dataset: {missing_columns}.")

label_encoders = {}
for column in ['Name', 'State', 'Type', 'BestTimeToVisit']:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le

if df['Popularity'].dtype == 'object':
    popularity_encoder = LabelEncoder()
    df['Popularity'] = popularity_encoder.fit_transform(df['Popularity'])
else:
    popularity_encoder = None

# Train the model
X = df[['State', 'Type', 'BestTimeToVisit', 'Popularity']]
y = df['Name']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    # Get user input
    user_input = request.json
    state_input = user_input.get('state').title()
    type_input = user_input.get('type').title()
    time_input = user_input.get('best_time').title()
    popularity_input = user_input.get('popularity')

    # Validate and encode input
    user_input_data = {}
    for column, value in zip(['State', 'Type', 'BestTimeToVisit'], [state_input, type_input, time_input]):
        if value not in label_encoders[column].classes_:
            return jsonify({"error": f"Invalid input for {column}: {value}. Valid options: {list(label_encoders[column].classes_)}"}), 400
        user_input_data[column] = label_encoders[column].transform([value])[0]

    if popularity_encoder:
        if popularity_input not in popularity_encoder.classes_:
            return jsonify({"error": f"Invalid input for Popularity: {popularity_input}. Valid options: {list(popularity_encoder.classes_)}"}), 400
        user_input_data['Popularity'] = popularity_encoder.transform([popularity_input])[0]
    else:
        try:
            user_input_data['Popularity'] = float(popularity_input)
        except ValueError:
            return jsonify({"error": "Popularity must be a valid numeric value."}), 400

    # Predict destination
    user_input_df = pd.DataFrame([user_input_data])
    predicted_index = model.predict(user_input_df)[0]
    predicted_destination = label_encoders['Name'].inverse_transform([predicted_index])[0]

    # Return recommendation
    return jsonify({
        "destination": predicted_destination,
        "image_url": f"/static/images/{predicted_destination}.jpg"  # Add image mapping
    })

if __name__ == '__main__':
    app.run(debug=True)
