import nltk
import re
import string
import inflect

def remove_punctuation(sen):
    regex = re.compile("[{}]".format(re.escape(string.punctuation)))
    return regex.sub("", sen)

def replace_numbers(sen_tok):
    """Substitues numeric numbers to text"""
    p = inflect.engine()
    new_tok = []
    for word in sen_tok:
        if word.isdigit():
            new_tok.append(p.number_to_words(word))
        else:
            new_tok.append(word)

    return new_tok

def lemmatize(sen_tok):
    lemmatizer = nltk.stem.WordNetLemmatizer()
    new_tok = []
    for word in sen_tok:
        new_tok.append(lemmatizer.lemmatize(word, pos='v'))
    return new_tok

def stem(sen_tok):
    stemmer = nltk.stem.SnowballStemmer("english")
    new_tok = []
    for word in sen_tok:
        new_tok.append(stemmer.stem(word))
    return new_tok

def remove_stopwords(sen_tok):
    new_tok = []
    for word in sen_tok:
        if word not in nltk.corpus.stopwords.words('english'):
            new_tok.append(word)

    return new_tok

def preprocess(sen):
    sen = sen.lower()
    sen = remove_punctuation(sen)
    sen_tok = nltk.word_tokenize(sen)
    sen_tok = replace_numbers(sen_tok)
    sen_tok = remove_stopwords(sen_tok)
    sen_tok = lemmatize(sen_tok)
    sen_tok = stem(sen_tok)
    return ' '.join(word for word in sen_tok)

def preprocessTFIDF(sen):
    sen = sen.lower()
    sen = remove_punctuation(sen)
    sen_tok = nltk.word_tokenize(sen)
    sen_tok = lemmatize(sen_tok)
    sen_tok = stem(sen_tok)
    return ' '.join(word for word in sen_tok)
