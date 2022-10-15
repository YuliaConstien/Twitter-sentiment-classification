# -*- coding: utf-8 -*-
"""
   Deep Learning for NLP
   Assignment 1: Sentiment Classification on a Feed-Forward Neural Network using Pretrained Embeddings
   Remember to use PyTorch for your NN implementation.
   Original code by Hande Celikkanat & Miikka Silfverberg. Minor modifications by Sharid Loáiciga.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gensim
import os

import string
import re




# Add the path to these data manipulation files if necessary:
# import sys
# sys.path.append('</PATH/TO/DATA/MANIP/FILES>')
from data_semeval import *
from paths import data_dir, model_dir


# name of the embeddings file to use
# Alternatively, you can also use the text file GoogleNews-pruned2tweets.txt (from Moodle),
# or the full set, wiz. GoogleNews-vectors-negative300.bin (from https://code.google.com/archive/p/word2vec/) 
embeddings_file = 'GoogleNews-pruned2tweets.bin'


#--- hyperparameters ---

# Feel free to experiment with different hyperparameters to see how they compare! 
# You can turn in your assignment with the best settings you find.

n_classes = len(LABEL_INDICES)
n_epochs = 30 
learning_rate = 0.001
report_every = 1
verbose = False

hidden_size = 64





#--- auxilary functions ---

# To convert string label to pytorch format:
def label_to_idx(label):
  return torch.LongTensor([LABEL_INDICES[label]])


#--- model ---

class FFNN(nn.Module):
  # Feel free to add whichever arguments you like here.
  # Note that pretrained_embeds is a numpy matrix of shape (num_embeddings, embedding_dim)
  def __init__(self, pretrained_embeds, n_classes, hidden_size):

      super(FFNN, self).__init__()

      # WRITE CODE HERE

      self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(pretrained_embeds))
      self.l1= nn.Linear(self.embedding.embedding_dim, hidden_size)
      self.relu1 = nn.ReLU()
      self.l2 = nn.Linear(hidden_size, n_classes)
      self.relu2= nn.ReLU()
      self.LogSoftmax = nn.LogSoftmax(dim=-1)

      #pass

  def forward(self, x):

    x = x.view(x.shape[0], -1)
    x = sum(self.embedding(x))
    x = self.l1(x)
    x= self.relu1(x)
    x= self.l2(x)
    x = self.relu2(x)

    return self.LogSoftmax(x)



    # WRITE CODE HERE
      #pass


#--- "main" ---

if __name__=='__main__':
  #--- data loading ---
  data = read_semeval_datasets(data_dir)
  gensim_embeds = gensim.models.KeyedVectors.load_word2vec_format(os.path.join(model_dir, embeddings_file), binary=True)
  pretrained_embeds = gensim_embeds.vectors
  # To convert words in the input tweet to indices of the embeddings matrix:
  word_to_idx = {word: i for i, word in enumerate(gensim_embeds.vocab.keys())}


  #--- set up ---
  # WRITE CODE HERE


  def word_2idx (data):
    w_in_idx = []
    for token in data :
      if token in word_to_idx:
        w_in_idx.append(word_to_idx[token])


    return torch.tensor(w_in_idx, dtype=torch.long)



  model = FFNN(pretrained_embeds, hidden_size, n_classes)
  loss_function = nn.NLLLoss()
  optimizer = optim.SGD(model.parameters(), lr=0.001)


  #--- training ---
  for epoch in range(n_epochs):
    total_loss = 0
    for tweet in data['training']:  
      gold_class = label_to_idx(tweet['SENTIMENT'])


      w_in_idx = word_2idx(tweet['BODY'])
      
      # WRITE CODE HERE

      # Forward pass to get output
      train_prediction = model(w_in_idx)


      # Calculate Loss
      train_loss = loss_function(train_prediction, gold_class)

      # Clearing up accumulated gradients
      optimizer.zero_grad()

      # Getting gradients
      train_loss.backward()

      # Updating parameters
      optimizer.step()

      # Add each mini batch's loss to the total loss
      total_loss += train_loss.item()



    if ((epoch+1) % report_every) == 0:
      print('epoch: %d, loss: %.4f' % (epoch, total_loss*100/len(data['training'])))
    
  # Feel free to use the development data to tune hyperparameters if you like!

  #--- test ---
  correct = 0
  with torch.no_grad():
    for tweet in data['test.gold']:
      gold_class = label_to_idx(tweet['SENTIMENT'])
    
      # WRITE CODE HERE


      w_in_idx = word_2idx(tweet['BODY'])
      #test_prediction = model(w_in_idx)


      predicted = model(w_in_idx)
      correct += torch.eq(predicted.exp().argmax(), gold_class).item()




      if verbose:
        print('TEST DATA: %s, OUTPUT: %s, GOLD LABEL: %d' % 
              (tweet['BODY'], tweet['SENTIMENT'], predicted))
        
    print('test accuracy: %.2f' % (100.0 * correct / len(data['test.gold'])))
