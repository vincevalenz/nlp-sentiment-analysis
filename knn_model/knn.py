import pickle
import pandas as pd
import matplotlib.pyplot as plt
import pathlib as p
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, plot_confusion_matrix, classification_report
from sklearn.utils import resample

############################################
# data preparation
###########################################

tr = open(p.Path.cwd().joinpath('datasets', 'emotion', 'train.txt')).read().split('\n')
val = open(p.Path.cwd().joinpath('datasets', 'emotion', 'val.txt')).read().split('\n')
ts = open(p.Path.cwd().joinpath('datasets', 'emotion', 'test.txt')).read().split('\n')

stopWords = set(stopwords.words('english'))
# Remove these words from the stopwords list that may be important to classifying emotion
stopWords = stopWords - {'not', 'dont', 'doesnt', 'didnt', 'no', 'down', 'very', 'under', 'below', 'off', 'on', 'up'}

# Separate data by each sample
train = [d.split(';') for d in tr]
train = train[:-1]

# Join words where apostrophe was replaced by a space
for s in train:
    if " t " in s[0]:
        s[0] = s[0].replace(" t ", "t ")

validation = [d.split(';') for d in val]
validation = validation[:-1]

for s in validation:
    if " t " in s[0]:
        s[0] = s[0].replace(" t ", "t ")

test = [d.split(';') for d in ts]
test = test[:-1]

for s in test:
    if " t " in s[0]:
        s[0] = s[0].replace(" t ", "t ")

# Create DataFrames for verification
train_df = pd.DataFrame(data=train + validation, columns=['doc', 'class'])
test_df = pd.DataFrame(data=test, columns=['doc', 'class'])

print("Train value counts:\n", train_df['class'].value_counts(),"\n")

##############################################
# balance data
##############################################

# separate each document by class
anger = train_df[train_df['class'] == 'anger']
fear = train_df[train_df['class'] == 'fear']
sadness = train_df[train_df['class'] == 'sadness']
joy = train_df[train_df['class'] == 'joy']
love = train_df[train_df['class'] == 'love']

# resample the minority classes
anger = resample(anger, replace=True, n_samples=3500)
fear = resample(fear, replace=True, n_samples=3500)
love = resample(love, replace=True, n_samples=6000)

# Put original and resampled data in single dataframe, excluding 'surprise' class
train_df = pd.concat([joy, love, anger, fear, sadness])


test_df = test_df[test_df['class'] != 'surprise']

train_df['class'].replace(['sadness', 'anger', 'fear'], 'unpleasant', inplace=True)
train_df['class'].replace(['love', 'joy'], 'pleasant', inplace=True)

test_df['class'].replace(['sadness', 'anger', 'fear'], 'unpleasant', inplace=True)
test_df['class'].replace(['love', 'joy'], 'pleasant', inplace=True)

print("Train value counts:\n", train_df['class'].value_counts(),"\n")
print("Test value counts:\n", test_df['class'].value_counts(),"\n")

print("train shape: ", train_df.shape)

##################################################
# Train model
##################################################

# TFIDF
vectorizer = TfidfVectorizer(analyzer='word', stop_words=stopWords)
tfidf_train = vectorizer.fit_transform(train_df['doc'].values.tolist())

# Initiate KNN classifier
knn_classifier = KNeighborsClassifier(n_neighbors=28, weights='distance', p=2, n_jobs=-1)
knn_classifier.fit(X=tfidf_train, y=train_df['class'])

# TFIDF on test data
tfidf_test = vectorizer.transform(test_df['doc'].values.tolist())

# test true classes
test_classes = test_df['class'].values.tolist()
# KNN predict on test
test_pred_knn = knn_classifier.predict(tfidf_test)

print("\n---KNN Results---\n")

print(classification_report(y_true=test_classes, y_pred=test_pred_knn))
print("test confusion matrix\n", confusion_matrix(y_true=test_classes, y_pred=test_pred_knn))
plot_confusion_matrix(estimator=knn_classifier, X=tfidf_test, y_true=test_classes)
plt.show()

##################################################
# Save model
##################################################

# UNCOMMENT TO SAVE NEW MODEL AND COUNTER

# model_filename = str(p.Path.cwd().joinpath('saved_models', 'knn_classifier_saved_model.sav'))
# pickle.dump(knn_classifier, open(model_filename, 'wb'))
#
# vectorizer_filename = str(p.Path.cwd().joinpath('saved_models', 'knn_tfidfvectorizer_saved_counter.sav'))
# pickle.dump(vectorizer, open(vectorizer_filename, 'wb'))
