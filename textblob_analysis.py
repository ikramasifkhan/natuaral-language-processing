from textblob import TextBlob
import string
import matplotlib.pyplot as plt


def percentage(part, whole):
    return 100 * float(part) / float(whole)


text = open('read.txt', encoding='utf-8').read()
lower_case_text = text.lower()

cleaned_text = lower_case_text.translate(str.maketrans('', '', string.punctuation))

blob = TextBlob(cleaned_text)

analysis = blob.sentiment.polarity

print(analysis)

if (analysis == 0):  # adding reaction of how people are reacting to find average later
    print("Neutral")
elif (analysis > 0 and analysis <= 0.3):
    print("Wpositive")
elif (analysis > 0.3 and analysis <= 0.6):
    print("Positive")
elif (analysis > 0.6 and analysis <= 1):
    print("Spositive")
elif (analysis > -0.3 and analysis <= 0):
    print("Wnegative")
elif (analysis > -0.6 and analysis <= -0.3):
    print("Negative")
elif (analysis > -1 and analysis <= -0.6):
    print("Snegative")




