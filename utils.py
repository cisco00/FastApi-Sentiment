import string
import nltk

from sklearn.base import TransformerMixin
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize

nltk.download("english")
nltk.download("wordnet")
nltk.download("puckt")

word_lemmitizer =WordNetLemmatizer()

def clean_text(text:str):
    #remove upper case
    text = text.lower()

    #removes puntuation from group of sentence
    for char in string.punctuation:
        text = text.replace(char, " ")


    #lemmitizer the words and join back into string text
    text = ' '.join(word_lemmitizer.lemmatize(word) for word in word_tokenize(text))
    return text

class DenseTransformer(TransformerMixin):
    def fit(self, x, y=None, **fit_params):
        return self

    @staticmethod
    def transform(x, y=None, **fit_param):
        return x.todense()

    def __self__(self):
        return DenseTransformer

    def __repr__(self):
        return self.__str__()
