# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.4.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% colab={} colab_type="code" id="eu7iP06fGljZ"
import tensorflow as tf
from transformers import (
    TFDistilBertForSequenceClassification,
)
from transformers.tokenization_distilbert import DistilBertTokenizerFast

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split

from utils import MAX_LEN

import os
import json
import spacy

spacy_nlp = spacy.load('en_core_web_sm')

BERTMODEL = "distilbert-base-cased"
PATH_CACHED_OUTPUT = '.cache/distilbert_outputs.json'
BATCH_SIZE = 32
N_EPOCHS = 30
LEARNING_RATE = 3e-5


class BertClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, max_seq_len=MAX_LEN, batch_size=BATCH_SIZE,
                 n_epochs=N_EPOCHS, val_size=0.1,
                 learning_rate=LEARNING_RATE):

        self.max_seq_len = max_seq_len
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.val_size = val_size
        self.learning_rate = learning_rate
        self.cached_output = None

        # Load dataset, tokenizer, model from pretrained model/vocabulary
        self.tokenizer = (
            DistilBertTokenizerFast
            .from_pretrained(BERTMODEL, do_lower_case=False)
        )
        self.model = (
            TFDistilBertForSequenceClassification
            .from_pretrained(BERTMODEL)
        )

        # Overwrite the 'distilbert' layer
        self.model.distilbert = self._distilbert_wrapper(self.model.distilbert)

    def _distilbert_wrapper(self, distilbert):

        if self.cached_output is None:
            if os.path.isfile(PATH_CACHED_OUTPUT):
                with open(PATH_CACHED_OUTPUT) as f:
                    self.cached_output = json.load(f)
            else:
                self.cached_output = {}

        def _distilbert(inputs, **kwargs):
            t_inputs = tuple(inputs)
            if t_inputs in self.cached_output:
                distilbert_output = self.cached_output[t_inputs].values
            else:
                distilbert_output = self.model.distilbert(inputs, **kwargs)
                self.cached_output[t_inputs] = distilbert_output

            return distilbert_output
        return _distilbert

    def summary(self):
        # Print model summary
        self.model.summary()

    def tokenize_sentences(self, sentences):
        # `encode_plus` will:
        #   (1) Tokenize the sentence.
        #   (2) Prepend the `[CLS]` token to the start.
        #   (3) Append the `[SEP]` token to the end.
        #   (4) Map tokens to their IDs.
        #   (5) Pad or truncate the sentence to `max_length`
        #   (6) Create attention masks for [PAD] tokens.
        encoded_dict = (
            self.tokenizer.batch_encode_plus(
                sentences,                   # Sentence to encode.
                add_special_tokens=True,     # Add '[CLS]' and '[SEP]'
                max_length=MAX_LEN,          # Pad & truncate all sentences.
                pad_to_max_length=True,
                return_attention_mask=True,  # Construct attn. masks.
                return_tensors='tf',
            )
        )

        return encoded_dict['input_ids'], encoded_dict['attention_mask']

    # set up the epoches to have better accuracy
    def train(self, X_train, y_train, X_val, y_val):
        self.scores = []
        self.loss = []
        num_batch = int(X_train.shape[0] / self.batch_size)
        for i in range(num_batch-1):
            start = i * self.batch_size
            end = (i + 1) * self.batch_size
            history = self.model.fit(X_train[start:end], y_train[start:end],
                                     validation_data=(X_val, y_val),
                                     epochs=self.n_epochs)
            self.scores.append(history.history['val_accuracy'])
            self.loss.append(history.history['val_loss'])

    def fit(self, X, y):
        X_train, X_val, y_train, y_val = (
            train_test_split(X, y, test_size=self.val_size)
        )

        ds_train = tf.data.Dataset.from_tensors(
            (self.tokenize_sentences(X_train.tolist()), y_train)
        ).shuffle(buffer_size=1024).batch(self.batch_size)

        ds_val = tf.data.Dataset.from_tensors(
            (self.tokenize_sentences(X_val.tolist()), y_val)
        ).batch(64)

        self.model.compile(optimizer='adam', loss='binary_crossentropy',
                           metrics=['accuracy'])
        # self.train(X_train, y_train, X_val, y_val)

        self.model.fit(ds_train, epochs=self.n_epochs,
                       steps_per_epoch=115,
                       validation_data=ds_val,
                       validation_steps=7)

        with open(PATH_CACHED_OUTPUT, 'w') as f:
            json.dump(self.cached_output, f)
