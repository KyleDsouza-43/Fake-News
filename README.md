# This project is about a Fake a News Detection system
# Author: Kyle Dsouza
# About Project:
# This project uses machine learning algorithm to check whether a given news headline is fake or not
# How it works?
# - Data Loading: Loads a dataset (100_news.csv) containing news articles and their labels (REAL or FAKE).
# - Text Preprocessing:
# - Converts text to lowercase.
# - Tokenizes using regular expressions (RegexpTokenizer).
# - Removes English stopwords.
# - Applies lemmatization using WordNetLemmatizer.
# - Feature Extraction:
# - Transforms cleaned text into numerical features using TfidfVectorizer with a limit of 5000 features.
# - Model Training:
# - Splits the data into training and testing sets (80/20).
# - Trains a LogisticRegression model on the TF-IDF features.
# - Model Evaluation:
# - Predicts labels on the test set.
# - Calculates accuracy score.
# - Displays a confusion matrix using seaborn for visual performance analysis.
# - Interactive Prediction:
# - Accepts user input via command line.
# - Preprocesses and vectorizes the input.
# - Predicts whether the news is REAL or FAKE using the trained model.