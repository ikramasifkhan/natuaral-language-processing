# This Program Created BY Ali Khan Bangladeshi Hacker
# Call Me If You Face Any problem +8801903800911
# FB : https://www.facebook.com/akwebsec.tk
import nltk
from nltk.corpus import movie_reviews
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import random
from nltk.classify.scikitlearn import SklearnClassifier
from textblob import TextBlob
from sklearn.naive_bayes import MultinomialNB
from textblob.classifiers import NaiveBayesClassifier

with open('reviews.txt', 'r', encoding='utf-8') as coders:
    hacker = str(coders.read()).split()

documents = [(list(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]

random.shuffle(documents)

training_set = documents
testing_set = documents

all_words = []

for words_list, categ in documents:
    for w in words_list:
        all_words.append(w.lower())

all_words = nltk.FreqDist(all_words)

word_features = list(hacker)


# print(word_features)

def find_features(training_set):
    words = set(training_set)
    features = {}
    for w in word_features:
        features[w] = (w in words)

    return features


featuresets = [(find_features(rev), category) for (rev, category) in documents]

random.shuffle(featuresets)

training_set = featuresets
testing_set = featuresets

MNB_classifier = SklearnClassifier(MultinomialNB())
classifier = nltk.NaiveBayesClassifier.train(training_set)

MNB_classifier.train(training_set)

with open('reviews.txt', 'r', encoding='utf-8') as coders:
    cracker = str(coders.read())


def sentiment_scores(sentence):
    sid_obj = SentimentIntensityAnalyzer()
    sentiment_dict = sid_obj.polarity_scores(sentence)
    return [sentiment_dict['neg'] * 100, sentiment_dict['pos'] * 100]


sentence = cracker
c1 = TextBlob(sentence)
print("Data Accuracy : ", (nltk.classify.accuracy(MNB_classifier, testing_set)) * 100)
print("TextBlob Polarity Result : ", (c1.polarity) * 100)
print("Negative Result Percent ", sentiment_scores(sentence)[0])
print("Positive Result Percent ", sentiment_scores(sentence)[1])