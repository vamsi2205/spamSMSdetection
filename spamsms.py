#import pandas and numpy
import pandas as pd
import numpy as np

#importing libraries which help to clean data
import re
import nltk
import string

#reading our file
dataset = pd.read_csv("spam.csv",encoding='latin-1')
#new column named 'message'
dataset["message"] = dataset["v2"]

#new column named 'labels'
dataset["labels"] = dataset["v1"].map({"ham":0,
                                          "spam":1})

#new data frame with only messages and labels
data = dataset[["labels","message"]]

import seaborn as sbn
import matplotlib.pyplot as plt

"""
#countplot for spam vs ham as imbalanced dataset
plt.figure(figsize = (8,8))
g = sbn.countplot(x = "labels", data = data)
p = plt.title("Countplot for spam vs ham")
p = plt.xlabel("Spam SMS")
p = plt.ylabel("Count")
plt.show()
"""

#Handling imbalanced data by oversampling
only_spam = data[data["labels"] == 1]

#print("No. of spam messages: ", len(only_spam))
#print("No. of ham messages: ",len(dataset) - len(only_spam))

#balancing the imbalanced data
count = int((data.shape[0] - only_spam.shape[0])/only_spam.shape[0])

for i in range (0, count-1):
    data = pd.concat([data , only_spam])


#countplot for spam vs ham as balanced dataset
plt.figure(figsize = (8,8))
g = sbn.countplot(x = "labels", data = data)
p = plt.title("Countplot for spam vs ham")
p = plt.xlabel("Spam SMS")
p = plt.ylabel("Count")
plt.show()


"""
#creating new feature wordcount
data["word_count"] = data["message"].apply(lambda x: len(x.split()))

plt.figure(figsize = (12,6))
#subplot 1
plt.subplot(1,2,1)
g = sbn.histplot(data[data["labels"] == 0].word_count, kde = True)
p = plt.title("Distribution of word_count for Ham SMS")

#subplot 2
plt.subplot(1,2,2)
g = sbn.histplot(data[data["labels"] == 1].word_count, kde = True)
p = plt.title("Distribution of word_count for Spam SMS")
plt.tight_layout()
plt.show()
"""

#new function to check for currency symbols
def is_currency(data):
    currency_symbols = ["$", "€", "£", "¥", "₹"]
    for i in currency_symbols:
        if i in data:
            return 1
    return 0

data["currency_symbols_present"] = data["message"].apply(is_currency)

"""
#countplot for currency_symbols_present
plt.figure(figsize = (8,8))
g = sbn.countplot(x = "currency_symbols_present", data = data, hue = "labels")
p = plt.title("Count Plot for currency symbols present")
p = plt.xlabel("Does SMS contain currency symbols")
p = plt.ylabel("Count")
p = plt.legend(labels = ["Ham", "Spam"], loc = 9)
plt.show()
"""

#Creating new feature containing numbers
def is_numbers(data):
    for i in data:
        if ord(i) >= 48 and ord(i) <= 57:
            return 1
    return 0

data["numbers_present"] = data["message"].apply(is_numbers)

"""
#countplot for currency_symbols_present
plt.figure(figsize = (8,8))
g = sbn.countplot(x = "numbers_present", data = data, hue = "labels")
p = plt.title("Count Plot for numbers present")
p = plt.xlabel("Does SMS contain numbers")
p = plt.ylabel("Count")
p = plt.legend(labels = ["Ham", "Spam"], loc = 9)
plt.show()
"""
#DATA CLEANING

#importing stopwords
from nltk.corpus import stopwords

#importing stemmer
from nltk.stem import WordNetLemmatizer
corpus = []
wnl = WordNetLemmatizer()

for sms in list(data.message):
    message = re.sub(pattern = "[^a-zA-z]", repl = " ", string = sms) #filtering out special characters
    message = message.lower()
    words = message.split() #Tokenizer
    filtered_words = [word for word in words if word not in set(stopwords.words("english"))]
    lemm_words = [wnl.lemmatize(word) for word in filtered_words]
    message = " ".join(lemm_words)

    corpus.append(message)

#creating the Bag of Words model
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(max_features = 500)
vectors = tfidf.fit_transform(corpus).toarray()
feature_names = tfidf.get_feature_names_out()

x = pd.DataFrame(vectors, columns = feature_names)
y = data["labels"]

from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import classification_report, confusion_matrix

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)


# #Naive Bayes model
# from sklearn.naive_bayes import MultinomialNB
# mnb = MultinomialNB()
# cv = cross_val_score(mnb, x, y, scoring = "f1", cv = 10)
# #print(round(cv.mean(),3))
# #print(round(cv.std(),3))

# mnb.fit(x_train, y_train)
# y_pred = mnb.predict(x_test)

# #print(classification_report(y_test, y_pred))

# cm = confusion_matrix(y_test, y_pred)
# plt.figure(figsize = (8,8))
# axis_labels = ["Ham", "Spam"]
# g = sbn.heatmap(data = cm, xticklabels = axis_labels, yticklabels = axis_labels, annot = True, fmt = "g", cbar_kws = {"shrink": 0.5})
# p = plt.title("Confusion Matrix of Multinomial Naive Bayes Theorem")
# p = plt.xlabel("Actual Values")
# p = plt.ylabel("Predicted Values")
# plt.show()


from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()
cv = cross_val_score(dt, x, y, scoring = "f1", cv = 10)
#print(round(cv.mean(),3))
#print(round(cv.std(),3))
dt.fit(x_train, y_train)
y_pred = dt.predict(x_test)
cm = confusion_matrix(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize = (8,8))
axis_labels = ["Ham", "Spam"]
g = sbn.heatmap(data = cm, xticklabels = axis_labels, yticklabels = axis_labels, annot = True, fmt = "g", cbar_kws = {"shrink": 0.5})
p = plt.title("Confusion Matrix of DT Classifier")
p = plt.xlabel("Actual Values")
p = plt.ylabel("Predicted Values")
plt.show()

def predict_spam(sms):
    message = re.sub(pattern = "[^a-zA-z]", repl = " ", string = sms) #filtering out special characters
    message = message.lower()
    words = message.split() #Tokenizer
    filtered_words = [word for word in words if word not in set(stopwords.words("english"))]
    lemm_words = [wnl.lemmatize(word) for word in filtered_words]
    message = " ".join(lemm_words)
    temp = tfidf.transform([message]).toarray()
    return temp

sample = input("Enter the SMS:")
sample = predict_spam(sample)
if dt.predict(sample):
    print("This is a spam message!") #DT classifier
else:
    print("This is a normal message!")


# if mnb.predict(sample):
#     print("This is a spam message!") #MNB classifier
# else:
#     print("This is a normal message!")
