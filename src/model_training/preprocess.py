"""Reusable preprocessing script for cleaning restaurant review text data."""
import re

import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer


nltk.download("stopwords")

ps = PorterStemmer()
all_stopwords = stopwords.words("english")
all_stopwords.remove("not")


def clean_review(review):
    """Clean review text.

    Parameters
    ----------
    review : str
        Review text to be cleaned.

    Returns
    -------
    review : str
        Cleaned review text.
    """
    review = re.sub("[^a-zA-Z]", " ", review)
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if word not in set(all_stopwords)]
    review = " ".join(review)
    return review
