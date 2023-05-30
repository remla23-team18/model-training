"""Training pipeline for the sentiment analysis model."""
import logging
import pickle
from pathlib import Path

import click
import joblib
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

from .preprocess import clean_review
from .preprocess import setup_stopwords


logger = logging.getLogger(__name__)


@click.command(name="train")
@click.option(
    "--output-dir",
    default="models",
    help="Directory where the model should be saved.",
)
@click.option(
    "--count-vectorizer-artifact-name",
    default="c1_BoW_Sentiment_Model.pkl",
    help="Name of the count vectorizer artifact.",
)
@click.option(
    "--classifier-artifact-name",
    default="c2_Classifier_Sentiment_Model",
    help="Name of the classifier artifact.",
)
def train_model(
    output_dir: str = "models",
    count_vectorizer_artifact_name: str = "c1_BoW_Sentiment_Model.pkl",
    classifier_artifact_name: str = "c2_Classifier_Sentiment_Model",
) -> None:
    """Train the sentiment analysis model."""
    dataset = pd.read_csv(
        "a1_RestaurantReviews_HistoricDump.tsv", delimiter="\t", quoting=3
    )

    corpus = []
    logger.debug("Cleaning and tokenizing reviews...")
    all_stopwords = setup_stopwords()
    for i in range(0, dataset["Review"].size):
        review = clean_review(dataset["Review"][i], all_stopwords)
        corpus.append(review)

    logger.debug("Creating the Bag of Words model...")
    cv = CountVectorizer(max_features=1420)
    X = cv.fit_transform(corpus).toarray()
    y = dataset.iloc[:, -1].values

    logger.info("Splitting the dataset into the Training set and Test set...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=0
    )
    classifier = GaussianNB()
    logger.info("Fitting Naive Bayes to the Training set...")
    classifier.fit(X_train, y_train)

    logger.info("Saving the model...")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.debug("Saving the count vectorizer to later use in prediction")
    pickle.dump(cv, open(output_dir / count_vectorizer_artifact_name, "wb"))

    logger.debug("Saving the classifier to later use in prediction")
    joblib.dump(classifier, output_dir / classifier_artifact_name)

    # Model performance measure
    # y_pred = classifier.predict(X_test)

    # from sklearn.metrics import confusion_matrix, accuracy_score
    # cm = confusion_matrix(y_test, y_pred)
    # print(cm)

    # accuracy_score(y_test, y_pred)
