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
from transformers import (
    TFDistilBertForSequenceClassification,
)
from transformers.tokenization_distilbert import DistilBertTokenizerFast

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split

from utils import MAX_LEN
import numpy as np


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

        # Load dataset, tokenizer, model from pretrained model/vocabulary
        self.tokenizer = (
            DistilBertTokenizerFast
            .from_pretrained(BERTMODEL, do_lower_case=False)
        )
        self.model = (
            TFDistilBertForSequenceClassification
            .from_pretrained(BERTMODEL)
        )

        # Freeze distilbert layer
        self.model.distilbert.trainable = False

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
            )
        )
        return (
            np.array(encoded_dict['input_ids']),
            np.array(encoded_dict['attention_mask'])
        )

    def train(self, X_train, y_train, X_val, y_val):
        self.scores = []
        self.loss = []
        num_batch = int(X_train[0].shape[0] / self.batch_size)
        for i in range(num_batch-1):
            start = i * self.batch_size
            end = (i + 1) * self.batch_size
            X_tr_dict = {'input_ids': X_train[0][start:end],
                         'attention_mask': X_train[1][start:end]}

            X_val_dict = {'input_ids': X_val[0],
                          'attention_mask': X_val[1]}

            print([v.shape for k, v in X_tr_dict.items()])
            print([v.shape for k, v in X_val_dict.items()])
            print(y_train[start:end].shape)

            history = self.model.fit(X_tr_dict, y_train[start:end],
                                     validation_data=(X_val_dict, y_val),
                                     epochs=self.n_epochs)
            self.scores.append(history.history['val_accuracy'])
            self.loss.append(history.history['val_loss'])

    def fit(self, X, y):
        X_train, X_val, y_train, y_val = (
            train_test_split(X, y, test_size=self.val_size)
        )

        X_train = self.tokenize_sentences(X_train.tolist())
        X_val = self.tokenize_sentences(X_val.tolist())

        self.model.compile(optimizer='adam', loss='binary_crossentropy',
                           metrics=['accuracy'])
        # self.train(X_train, y_train, X_val, y_val)

        self.train(X_train, one_hot_encoder(y_train),
                   X_val, one_hot_encoder(y_val))


def one_hot_encoder(y):
    one_hot = np.zeros((y.shape[0], 2))
    one_hot[y, 1] = 1
    return one_hot
