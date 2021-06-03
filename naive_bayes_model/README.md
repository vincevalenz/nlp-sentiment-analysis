# Naive Bayes Model

## Introduction

This folder contains the file `nb.py` that does all the data preprocessing and training of the naive bayes model.

The file has been commented explaining what each part of the code does, but I will further elaborate in this README.

## Packages


* Sci-Kit Learn
  * `pip install scikit-learn`
* NLTK
  * `pip install nltk`
* Pandas
  * `pip install pandas`
* Matplotlib
  * `pip install matplotlib`
* Pathlib
  * `pip install pathlib`
  
## Description

* ### Preprocess Data
    * This section reads in the data files `train.txt`, `validation.txt` and `test.txt` from folder `/datasets/emotion` as raw text where each sample is separated by `\n`. It then splits the data on `\n` and puts them into a list. 
    It additionally applys some preprocessing to the samples by iterating through them and joining words that have been split at the apostrophe. For example the word `don t` becomes `dont`. After completing the preprocessing the data is put into `pandas` dataframes where the train and validation sets are combined into `train_df` and the test set is in `test_df`.
    * Prepare the set of stopwords from the `nltk` corpus. Remove words that can negate such as 'not', 'do', 'does', etc...

* ### Balance Data
  * This section rebalances the dataset so that there are two classes, pleasant and unpleasant.
  * Remove class "surprise" from set as it does not fit into the emotion model
  * Group joy (6066) and love (1482) into "pleasant" class
      * Balance the subclasses by resamplimg love class to be of size 6000
  * Group classes sadness (5216), anger (2434) and fear (2149) into "unpleasant" category
      * Balance the subclasses by resampling anger to size 3500 and fear to size 3500
  * Classes unpleasant (12216) and pleasant (12066) are now about 50/50
  
* ### Train Model
  * This section uses the `MultinomialNB` classifier and the `CountVectorizer` to train the model on the dataset.
  * Initialize the count vectorizer with parameters
    * `analyzer='word'`
    * `stop_words=stopWords`
    * `ngram_range=(1, 2)`
  * Use the function `fit_transform` to train the count vectorizer on the training data
    * The data needs to be in a list format so I used `train_df['doc'].values.tolist()` to achieve this
  * Initialize the `MultinomialNB` classifier and use `fit` to train the model

  * Use the function `transform` from the count vectorizer to transform the test data
  * Use the function `predict` from the classifier on the transformed test data to generate the prediction results
  
* ### Saved Models
    * This section uses the `pickle` package built into python to save the model into the `/saved_models` directory
    * To save a new model, uncomment this section