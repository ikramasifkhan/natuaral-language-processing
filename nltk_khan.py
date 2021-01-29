# This Program Created BY Ali Khan Bangladeshi Hacker
# Call Me If You Face Any problem +8801903800911
# FB : https://www.facebook.com/akwebsec.tk
def NlTKProJect():
    import nltk
    import numpy as np
    from nltk.corpus import stopwords
    import string
    nltk.download('stopwords')
    from nltk.classify.scikitlearn import SklearnClassifier
    from nltk.classify import ClassifierI
    from nltk.tokenize import word_tokenize
    from nltk.corpus import movie_reviews
    from sklearn.naive_bayes import MultinomialNB

    with open('reviews.txt', 'r', encoding='utf-8') as coders:
        hacker = str(coders.read()).split()

    with open('reviews.txt', 'r', encoding='utf-8') as coders:
        cracker = str(coders.read())

    documents = [(list(movie_reviews.words(fileid)), category)
                 for category in movie_reviews.categories()
                 for fileid in movie_reviews.fileids(category)]

    np.random.shuffle(documents)

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

    np.random.shuffle(featuresets)

    training_set = featuresets
    testing_set = featuresets

    MNB_classifier = SklearnClassifier(MultinomialNB())
    classifier = nltk.NaiveBayesClassifier.train(training_set)

    MNB_classifier.train(training_set)
    print("Data Accuracy Result : ", (nltk.classify.accuracy(MNB_classifier, testing_set)) * 100)

    print(str("NLTK Classifier Accuracy Result : "), (nltk.classify.accuracy(classifier, testing_set)) * 100)

    stopwords_english = stopwords.words('english')

    def bag_of_words(words):
        words_clean = []

        for word in words:
            word = word.lower()
            if word not in stopwords_english and word not in string.punctuation:
                words_clean.append(word)

        words_dictionary = dict([word, True] for word in words_clean)

        return words_dictionary

    custom_review = cracker
    custom_review_tokens = word_tokenize(custom_review)
    custom_review_set = bag_of_words(custom_review_tokens)
    prob_result = classifier.prob_classify(custom_review_set)
    # print(prob_result.max())
    print("Negative Result Percent " + "{:.0%}".format(prob_result.prob("neg")))  # It is a negative
    print("Positive Result Percent " + "{:.0%}".format(prob_result.prob("pos")))  # It is a positive
    # classifier.show_most_informative_features(5)


if __name__ == '__main__':
    NlTKProJect()
