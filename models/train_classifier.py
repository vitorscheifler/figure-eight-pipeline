import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
import re
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, \
                            precision_score, recall_score, f1_score
import pickle
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


def load_data(database_filepath):
    """Read data from a sqlite database and split it, retunrning X and y """

    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('messages', engine)
    X = df['message']
    y = df.iloc[:, 4:]
    category_names = y.columns
    return X, y, category_names

def tokenize(text):
    """Returns a text without punctuation, case normilized, tokenized,
       stemmed and lemmatized.
    """
    text = re.sub(r'[^a-zA-Z0-9]', " ", text.lower())

    tokens = word_tokenize(text)

    tokens = [tk for tk in tokens if tk not in stopwords.words('english')]

    tokens = [WordNetLemmatizer().lemmatize(w) for w in tokens]

    return tokens

def build_model():
    """Create and return a tuned pipeline"""
    pipeline = Pipeline([('vect', CountVectorizer(tokenizer=tokenize)),
                         ('tfidf', TfidfTransformer()),
                         ('clf', MultiOutputClassifier(AdaBoostClassifier()))])

    params = {"clf__estimator__n_estimators" : [50, 100],
              "vect__ngram_range" : [(1, 1), (1, 2)]
              }

    cv = GridSearchCV(pipeline, param_grid = params, cv=3)
    return cv

def evaluate_model(model, X_test, Y_test, category_names):
    """Returns classification metrics for each class and the same metrics for the overall model"""
    Y_pred = model.predict(X_test)
    Y_pred = pd.DataFrame(Y_pred).reset_index(drop=True)
    Y_pred.columns = Y_test.columns
    for col, pred in zip(Y_test, Y_pred):
        print(col, classification_report(Y_test[col], Y_pred[pred]))

    print("Accuracy: ",
          accuracy_score(Y_test.values.reshape(-1,1),
                         Y_pred.values.reshape(-1,1)))
    print("F1 Score: ",
          f1_score(Y_test.values.reshape(-1,1),
                   Y_pred.values.reshape(-1,1)))
    print("Precision: ",
          precision_score(Y_test.values.reshape(-1,1),
                          Y_pred.values.reshape(-1,1)))
    print("Recall: ",
          recall_score(Y_test.values.reshape(-1,1),
                       Y_pred.values.reshape(-1,1)))

def save_model(model, model_filepath):
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
