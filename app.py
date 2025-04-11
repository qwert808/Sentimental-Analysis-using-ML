import os
import re
from flask import Flask, request, jsonify, render_template
from joblib import load

app = Flask(__name__)

# Get the directory where app.py is located
base_dir = os.path.dirname(os.path.abspath(__file__))

# Load the vectorizer and models
try:
    vectorizer_path = os.path.join(base_dir, "tfidf_vectorizer.pkl")
    if not os.path.exists(vectorizer_path):
        raise FileNotFoundError(f"Vectorizer file not found: {vectorizer_path}")
    vectorizer = load(vectorizer_path)

    models = {}
    model_files = {
        "Logistic Regression": "logistic_regression_model.pkl",
        "Naive Bayes": "multinomial_nb_model.pkl",
        "Random Forest": "random_forest_model.pkl",
        "SVM": "svm_model.pkl"
    }
    for model_name, model_file in model_files.items():
        model_path = os.path.join(base_dir, model_file)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        models[model_name] = load(model_path)
    print("Vectorizer and models loaded successfully!")
except Exception as e:
    print(f"Error loading models: {e}")
    raise

# Preprocessing function to handle negations
def preprocess_comment(comment):
    # Replace negations with combined words (e.g., "not good" -> "not_good")
    comment = re.sub(r'\b(not)\s+(\w+)', r'\1_\2', comment)
    return comment

# Split comment into phrases based on conjunctions/keywords
def split_into_phrases(comment):
    return re.split(r'\b(but|however|although|though)\b', comment, flags=re.IGNORECASE)

# Calculate weighted sentiment score for the comment
def calculate_weighted_sentiment(phrases):
    # Generate weights dynamically for the number of phrases
    weights = [i / sum(range(1, len(phrases) + 1)) for i in range(1, len(phrases) + 1)]

    sentiment_scores = []
    for phrase in phrases:
        # Preprocess and vectorize each phrase
        processed_phrase = preprocess_comment(phrase.strip())
        phrase_vectorized = vectorizer.transform([processed_phrase])

        # Predict sentiment for each model
        phrase_sentiments = []
        for model in models.values():
            prediction = model.predict(phrase_vectorized)[0]
            sentiment_score = 1 if prediction == 1 else -1  # Map positive to 1 and negative to -1
            phrase_sentiments.append(sentiment_score)

        # Use the average sentiment as the phrase's score
        sentiment_scores.append(sum(phrase_sentiments) / len(models))

    # Calculate the weighted sum of sentiment scores
    final_score = sum(w * s for w, s in zip(weights, sentiment_scores))
    return "positive" if final_score > 0 else "negative"

@app.route('/')
def home():
    # Render the HTML file for user interaction
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Parse the input JSON request
        data = request.get_json()

        # Extract the comment from the request
        comment = data.get('comment')

        # Validate input
        if not comment:
            return jsonify({"error": "No comment provided"}), 400

        # Split comment into phrases and calculate weighted sentiment
        phrases = split_into_phrases(comment)
        final_sentiment = calculate_weighted_sentiment(phrases)

        # Collect individual model predictions for transparency
        model_predictions = {}
        for model_name, model in models.items():
            processed_comment = preprocess_comment(comment)
            comment_vectorized = vectorizer.transform([processed_comment])
            prediction = model.predict(comment_vectorized)[0]
            sentiment = "positive" if prediction == 1 else "negative"
            model_predictions[model_name] = sentiment

        # Return the predictions and final sentiment
        return jsonify({
            "comment": comment,
            "individual_model_predictions": model_predictions,
            "final_sentiment": final_sentiment
        }), 200

    except Exception as e:
        # Handle unexpected errors
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
