
# About the code: 

This is a sentiment analysis problem statement solved using the dataset from kaggle.
The main aim was multiclass classification of tweets using NLP

Both Machine Learning and Deep Learning approaches were explored:
ML models include:
- Multinomial Naive Bayes Classifier, 
- Gradient Boosting classifier,
- Random Forest Classifier( relatively best training accuracy around 77%)
- As expected, they performed poorly on test data(only 35% accuracy)

Deep Learning approach includes:
A simple RNN model,(Accuracy: around 75%)
A LSTM based model,(Accuracy around 82%)
A Bidirectional LSTM model(Accuracy around 84% but suffers from overfitting)

# How To Use

This folder contains 3 files:-
1. Sentiment Analysis.ipynb
2. Corona_NLP_train.csv 
3. Corona_NLP_test.csv


## ------------------------About the module----------------------------------------------------------

- Tools used: Jupyter notebook in Ananconda environment
- Dependencies: Python 3, Tensorflow version 2.5.0, Keras, nltk
- Libraries used: Numpy, Sklearn, Seaborn, Keras, Tensorflow, Matplotlib, gensim

## ===================Instructions to run the code======================================

#### 1. IN JUPYTER NOTEBOOK:
- The folder contains the train and test data in form of .csv files ('Corona_NLP_train.csv' and 'Corona_NLP_test.csv')
- Ensure you download the whole folder and not change the relative path of .ipynb and .csv files.
- Run the code cells sequentially 
NOTE: Some models may take time to execute

#### 2. In GOOGLE COLABORATORY
- open the .ipynb file
- Upload both the .csv files using the file upload option( mostly available at left hand side menu bar)
- Ensure upload is completed
- Execute cells sequentially
NOTE: Some models may take time to execute





