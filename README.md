# Interactive NLP application for classifying pleasant and unpleasant emotion from text

### Author
&emsp;Vincent Valenzuela

## Introduction
&emsp;This repository contains all the components involved in building the interactive application. It includes the dataset,
classification models and model pipelines, previous saved models, relevant media and the application itself.
Each component is separated in its respective directory and more information can be found in the README files within those directories. 

&emsp;To run the application, navigate to the `/app` directory and follow the instructions in the `README.md` file.


## Directories
* `/app`: Main application
    * `main/py`
* `/datasets`: Emotion dataset
    * `/emotion`
        * `train.txt`
        * `validation.txt`
        * `test.txt`
* `/knn_model`: K nearest neighbors model
    * `knn.y`
    * `KNN_Hyperparameters.ipynb`
* `/media`: Relevant media
    * `pleasant_vid.mp4`
    * `unpleasant_vid.mp4`
* `/naive_bayes_model`: Naive Bayes model
    * `nb.py`
* `/saved_models`: Current trained models 