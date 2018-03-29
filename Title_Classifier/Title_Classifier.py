"""
Classify titles based on word commonality and string content
"""

import sqlite3
import re
import pandas as pd
import numpy as np
import nltk
import sklearn
from sklearn import model_selection, svm, neighbors
from sklearn.naive_bayes import MultinomialNB, BernoulliNB

# Variable fields to adapt to different data sets --------------------------------------------------------------

values_to_replace = {'clevel': 1, 'vplevel': 2, 'directorlevel': 3, 'managerlevel': 4, 'staff': 5}
level_field = 'Management Level'
title_field = 'Title'

# st.stem() sklearn's word stemming algorithm
st = nltk.stem.lancaster.LancasterStemmer()
# sklearn's vectorizor for bag of words analysis
count_vect = sklearn.feature_extraction.text.CountVectorizer()

# Import Data --------------------------------------------------------------------------------------------------

db = sqlite3.connect('EloquaDB.db')
contacts = pd.read_sql("""SELECT ContactID, "SFDC Account ID", "Email Address", "Email Address Domain",
                        Title, Department, "Management Level" FROM contacts;""", con=db)

try:
    title_abrev = pd.read_pickle('title_abrev.p')
except FileNotFoundError:
    title_abrev = pd.read_csv('title_abrev.csv', sep=',')
    # print(title_abrev)
    # title_abrev = title_abrev[0].rename(columns={0: 'Abrev', 1: 'Title'})
    title_abrev.Title = title_abrev.Title.str.split(',').str[0].str.lower()
    title_abrev.to_pickle('title_abrev.p')


def replace_str(series):
    """
    A short function to replace any abbreviations with their full form
    :param series: series to be cleansed
    :return: the cleansed series
    """
    print(series.tail())
    for a, t in zip(title_abrev.Abrev, title_abrev.Title):
        series = series.apply(lambda x: re.sub(r"\b" + re.escape(a) + r"\b", t, x, flags=re.IGNORECASE))
    print('Replaced')
    print(series.tail())
    return series


# r'\b{}\b'.format(a)  , flags=re.IGNORECASE
# Process Data --------------------------------------------------------------------------------------------------

def process(df):
    """
    The main processing stem for data inputs
    :param df: data frame to be cleansed
    :return: cleansed data frame
    """
    df[level_field] = df[level_field].str.lower().str.replace('[^\w\s]', '')
    df[level_field] = df[level_field].map(values_to_replace)
    df['TitleFormatted'] = df[title_field].str.lower().str.replace('head,', 'head of').str.replace('[^\w\s]', ' ')
    df['TitleFormatted'] = replace_str(df['TitleFormatted'])  # Use function to replace abbreviations and misspellings
    df['TitleFormatted'] = df['TitleFormatted'].apply(
        lambda row: ' '.join([st.stem(y) for y in row.split(" ")]))

    return df


print("Processing contact data.")
contacts = process(contacts)

# ---------------------------------------------------------------------------------------------------------------
# Split data set

data_set = contacts[pd.notnull(contacts[level_field])]

print("Transforming titles to vectors.")
X = count_vect.fit_transform(data_set['TitleFormatted'])
y = data_set[level_field]

# print(data_set.describe) # Describe the data_set
# print(X.shape)
# print(count_vect.vocabulary_.get(u'man'))
# print(y.shape)
# print(bag_of_words.vocabulary_.get('man')) # Print how many times a particular stem appears


# Splits the data into training and testing data sets (20%)
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.20)

# Select the classifier
clf = MultinomialNB()
print("Training MultinomialNB classifier.")
clf.fit(X_train, y_train)

# Return Accuracy score
accuracy = clf.score(X_test, y_test)
print("Estimated accuracy is: {}%".format(accuracy))

# Create a copy of original data set to analyze
contact_copy = contacts
X_process = count_vect.transform(contacts['TitleFormatted'])
prediction = clf.predict(X_process)
contact_copy[level_field+'_P'] = prediction

# Create a column that highlights where the prediction is different than the current value ----------------------------
contact_copy["Is_Same"] = np.where(contact_copy[level_field] == contact_copy[level_field+'_P'], 'yes', 'no')
# ---------------------------------------------------------------------------------------------------------------------

# drop working column and export to csv
# contact_copy.drop('TitleFormatted', axis=1, inplace=True)
print(contact_copy.describe())
print("Exporting predictions.")
contact_copy.to_csv('test_predictions.csv', sep='|')
