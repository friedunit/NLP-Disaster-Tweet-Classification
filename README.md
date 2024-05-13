# NLP-Disaster-Tweet-Classification

## Overview

#### This project focuses on Natural Language Processing which is part of machine learning and a way for computers to learn and analyze the human language. The Kaggle competition can be found here: https://www.kaggle.com/c/nlp-getting-started/overview and the data was downloaded from https://www.kaggle.com/c/nlp-getting-started/data

#### Twitter has become an important communication channel in times of emergency. The ubiquitousness of smartphones enables people to announce an emergency they’re observing in real-time. Because of this, more agencies are interested in programatically monitoring Twitter (i.e. disaster relief organizations and news agencies). For this challenege, we are building a machine learning model that predicts which Tweets are about real disasters and which one’s aren’t.

## Initial findings in the data:

* We can see from the initial training data EDA that there are 5 columns and 7,613 observations
* It looks like 'text' and 'target' are the features we'll focus on and can probably drop the other 3. There were also several null values in both 'keyword' and 'location'.
* Of the 7,613 'text' observations, 7,503 are unique so there must be 110 duplicates, we can take care of that
* More than half of the training tweets are NOT true disaster tweets
* We also have a test set of 4 columns (just missing the 'target' feature) and 3,263 observations

![image](https://github.com/friedunit/NLP-Disaster-Tweet-Classification/assets/10797098/cbc2190b-6080-4181-879a-3633061cc8b8)

#### After removing duplicates from the tweet dataframe 'text' column, 57% are now 0 or NON disaster tweets and 42% ARE classified as disaster tweets

## Text Cleaning & Preprocessing

#### In natural language processing, text needs to be cleaned and preprocessed before being fed into a model. Removing punctuations, stop words (such as 'I', 'is', 'the', etc), setting text to lowercase, port stemming (leaving the root word be removing tense) and also tokenizing or using a countvectorizer helps the model process and analyze text easier.

#### Below, I created the clean_text() function to take in each text line, set it to lowercase, remove punctuation, remove English stopwords, stem the words, then join as string for the return value. 

## Model Architecture

#### First, we will split the training data using sklearn's train_test_split into 80/20 training/test. For the model, I'll use sklearn's logistic regression since we are predicting a binary output (0 for NOT disaster tweet, 1 for disaster tweet)

## Model Evaluation

#### For the Kaggle competition, F1 scores are used to determine accuracy between predicted values and true values.

![image](https://github.com/friedunit/NLP-Disaster-Tweet-Classification/assets/10797098/6c2415a0-2fb6-4212-8631-c21fb5ec3e30)

## Findings and Conclusion

#### After making the predictions of the X_test data and comparing to the y_test 'target' data, the F1 Score for the model was about 73%. Not bad but could probably get better with other text cleaning or perhaps a different model for classification. We can also see in the confusion matrix above, that the model predicted 755 True Positives and 433 True Negatives correctly. For the incorrect predictions, there were 194 False Negatives and 119 False Positives.

