import os
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

TRAIN_PATH = os.path.join("data", "processed", "train.csv")
TEST_PATH = os.path.join("data", "processed", "test.csv")

# custom transformer to extract lightweight text metadata features
class TextStatsExtractor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        features = []
        for text in X:
            t = str(text)
            char_count = len(t)
            word_count = len(t.split())
            avg_word_len = char_count / (word_count + 1e-5)
            has_question = 1.0 if '?' in t else 0.0
            has_exclamation = 1.0 if '!' in t else 0.0
            features.append([char_count, word_count, avg_word_len, has_question, has_exclamation])
        return np.array(features)

def run_experiment():
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    X_train = train_df['text']
    y_train = train_df['label']
    X_test = test_df['text']
    y_test = test_df['label']

    feature_extractors = {
        "1. Word TF-IDF (1, 2)": TfidfVectorizer(
            analyzer='word', ngram_range=(1, 2), max_features=5000, stop_words='english'
        ),
        "2. Char TF-IDF (3, 5)": TfidfVectorizer(
            analyzer='char', ngram_range=(3, 5), max_features=5000
        ),
        "3. Word + Char TF-IDF": FeatureUnion([
            ('word_tfidf', TfidfVectorizer(analyzer='word', ngram_range=(1, 2), max_features=4000, stop_words='english')),
            ('char_tfidf', TfidfVectorizer(analyzer='char', ngram_range=(3, 5), max_features=4000))
        ]),
        "4. Word TF-IDF + Text Stats": FeatureUnion([
            ('word_tfidf', TfidfVectorizer(analyzer='word', ngram_range=(1, 2), max_features=5000, stop_words='english')),
            ('stats', TextStatsExtractor())
        ])
    }

    print(f"{'Feature Strategy':<30} | {'Features':>10} | {'Test Acc':>10} | {'Macro F1':>10}")
    print("-" * 67)

    for name, extractor in feature_extractors.items():
        pipeline = Pipeline([
            ('features', extractor),
            ('classifier', LogisticRegression(C=10, max_iter=1000, random_state=42))
        ])

        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_test)

        # Count total features
        X_train_trans = extractor.fit_transform(X_train)
        n_features = X_train_trans.shape[1]

        acc = accuracy_score(y_test, preds)
        macro_f1 = f1_score(y_test, preds, average='macro')

        print(f"{name:<30} | {n_features:>10} | {acc*100:>9.2f}% | {macro_f1*100:>9.2f}%")

if __name__ == "__main__":
    run_experiment()
