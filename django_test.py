from __future__ import print_function, division
from builtins import range, input
# Note: you may need to update your version of future
# sudo pip install -U future

import os, sys

from keras.models import Model
from keras.layers import Input, LSTM, GRU, Dense, Embedding
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt

import keras.backend as K
import pickle

with open('encoder_model.pickle','rb') as f:
    encoder_model = pickle.load(f)

with open('decoder_model.pickle','rb') as f:
    decoder_model = pickle.load(f)


with open('word2index_ouputs.pickle','rb') as f:
    word2idx_outputs = pickle.load(f)

with open('idx2word_trans.pickle','rb') as f:
    idx2word_trans = pickle.load(f)
    
with open('input_tokenizer.pickle','rb') as f:
    tokenizer_inputs = pickle.load(f)

max_len_input = 30
max_len_target = 33
def decode_sequence(input_seq):
  # Encode the input as state vectors.
  states_value = encoder_model.predict(input_seq)

  # Generate empty target sequence of length 1.
  target_seq = np.zeros((1, 1))

  # Populate the first character of target sequence with the start character.
  # NOTE: tokenizer lower-cases all words
  target_seq[0, 0] = word2idx_outputs['<sos>']

  # if we get this we break
  eos = word2idx_outputs['<eos>']

  # Create the translation
  output_sentence = []
  for _ in range(max_len_target):
    output_tokens, h, c = decoder_model.predict(
      [target_seq] + states_value
    )
    # output_tokens, h = decoder_model.predict(
    #     [target_seq] + states_value
    # ) # gru

    # Get next word
    idx = np.argmax(output_tokens[0, 0, :])

    # End sentence of EOS
    if eos == idx:
      break

    word = ''
    if idx > 0:
      word = idx2word_trans[idx]
      output_sentence.append(word)

    # Update the decoder input
    # which is just the word just generated
    target_seq[0, 0] = idx

    # Update states
    states_value = [h, c]
    # states_value = [h] # gru

  return ' '.join(output_sentence)

def preprocess(a): 
    test = tokenizer_inputs.texts_to_sequences([a])
    return(pad_sequences(test, maxlen=max_len_input))

