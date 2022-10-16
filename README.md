# Introduction

This repository is for one of the projects in the Deep Learning for Natural Language Processing course, in the master of Cognitive Systems programme of Potsdam University.

In this assignment,  the goal is to implement a sentiment classifier for tweets. The instructors make avialable: a set of
tweets, some starter code, and a size-wise pruned version of Google’s pre-trained embeddings.

# Code Structure

The directory of the assignment has the following structure: 

– data/

––– development.gold.txt

––– development.input.txt

––– test.gold.txt

––– test.input.txt

––– training.txt

–model/

––– GoogleNews-pruned2tweets.bin

– src/

––– data_semeval.py

––– assignment_CONSTIEN.py

––– paths.py

The folder includes a pruned version of the Google’s pretrained word embeddings, restricted
only to the words in the dataset (17M as oppposed to 1.5G).

To run the code the following is required: PyTorch, NumPy and NLTK (for tokenization). 

# Twitter Data

The provided data is a dataset from the 2014 SemEval shared task “Sentiment analysis in
Twitter”. For background information or information on systems which participated in the competition, please have a look at:

S. Rosenthal, A. Ritter, P. Nakov and V. Stoyanov. SemEval-2014 Task 9: Sentiment Analysis
in Twitter. SemEval 2014.

For more information on the data take a look at the data directory. Each data file consists
of lines with three fields: a Twitter ID number, a sentiment class (positive, neutral or negative),
and a tweet. 

The files development.input.txt and test.input.txt do not show the correct sentiment
label for the tweets. Instead, they have a dummy unknown label in the second field. The correct labels can be found at development.gold.txt and test.gold.txt.

The development dataset was used to fine-tune the hyperparameters of the model. The
test set is only used for the final evaluation run. 
