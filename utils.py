import re
import spacy
import nltk
import pandas as pd
import numpy as np
from unicodedata import normalize
from nltk.corpus import stopwords
from sklearn.base import BaseEstimator, TransformerMixin
from spacy.attrs import LEMMA, IS_ALPHA
from tqdm import tqdm

URL_RE = re.compile(r"http\S+")
MENTION_RE = re.compile(r"@\S+")
RT_RE = re.compile(r"RT @[\w_]+:")
NUM_RE = re.compile(r"\d+")
SPACE_RE = re.compile(r"\s+")

spacy_nlp = spacy.load(
    "pt_core_news_sm", 
    disable=["parser", "ner", "textcat", "attribute_ruler"]
)
spacy_nlp.max_length = 3_000_000

# --------------------------------------------------
# 2. GLOBAL CONSTANTS & REGEX
# --------------------------------------------------
try:
    pt_stopwords = set(stopwords.words("portuguese"))
except LookupError:
    nltk.download("stopwords")
    pt_stopwords = set(stopwords.words("portuguese"))


class SpacyPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, max_tokens=100, batch_size=1024, n_process=-1, show_progress=True):
        self.max_tokens = max_tokens
        self.batch_size = batch_size
        self.n_process = n_process
        self.show_progress = show_progress

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        vocab = spacy_nlp.vocab
        stop_ids = {vocab.strings[w] for w in pt_stopwords if w in vocab.strings}
        
        # Generator for parallel cleaning
        cleaned_gen = (clean_text(t) for t in X)
        
        total = len(X) if hasattr(X, '__len__') else None
        pipe_gen = spacy_nlp.pipe(
            cleaned_gen, 
            batch_size=self.batch_size, 
            n_process=self.n_process
        )
        
        if self.show_progress:
            pipe_gen = tqdm(pipe_gen, total=total, desc="Lemmatizing Portuguese Text")

        processed = []
        # Optimization: specifically enabling only what is strictly necessary
        with spacy_nlp.select_pipes(enable=["lemmatizer"]):
            for doc in pipe_gen:
                attr_array = doc.to_array([LEMMA, IS_ALPHA])
                
                if attr_array.size == 0:
                    processed.append("")
                    continue

                # Filtering using integer IDs (row[0]=LEMMA, row[1]=IS_ALPHA)
                lemmas = [
                    vocab.strings[row[0]] 
                    for row in attr_array 
                    if row[1] == 1 and row[0] not in stop_ids
                ]
                
                processed.append(" ".join(lemmas[:self.max_tokens]))
                # Periodically clear memory if processing millions of rows
                if len(processed) % 5000 == 0:
                    import gc
                    gc.collect()

        return processed


def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = URL_RE.sub(" ", text)
    text = MENTION_RE.sub(" ", text)
    text = RT_RE.sub(" ", text)
    text = NUM_RE.sub(" ", text)
    # Strip accents
    text = normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    return SPACE_RE.sub(" ", text).strip()
