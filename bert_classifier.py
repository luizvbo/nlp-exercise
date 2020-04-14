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
    DistilBertTokenizer,
)

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split

from utils import MAX_LEN
from tqdm import tqdm
import numpy as np

import spacy

spacy_nlp = spacy.load('en_core_web_sm')

BERTMODEL = "distilbert-base-uncased"
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
            DistilBertTokenizer
            .from_pretrained(BERTMODEL, cache_dir='./.cache/',
                             do_lower_case=True)
        )
        self.model = (
            TFDistilBertForSequenceClassification
            .from_pretrained(BERTMODEL, cache_dir='./.cache/')
        )

        # Freezes all the layers (except the classifier and dropout)
        print("Trainable layers")
        print("=" * 16)
        for layer in self.model.layers:
            frozen_layers = ('distilbert')
            layer.trainable = layer.name not in frozen_layers
            print(layer.name + ":",  layer.trainable)

    def summary(self):
        # Print model summary
        self.model.summary()

    def tokenize_sentences(self, sentences):
        input_ids = []
        attention_masks = []

        for sentence in tqdm(sentences):
            # `encode_plus` will:
            #   (1) Tokenize the sentence.
            #   (2) Prepend the `[CLS]` token to the start.
            #   (3) Append the `[SEP]` token to the end.
            #   (4) Map tokens to their IDs.
            #   (5) Pad or truncate the sentence to `max_length`
            #   (6) Create attention masks for [PAD] tokens.
            encoded_dict = (
                self.tokenizer.encode_plus(
                    sentence,                     # Sentence to encode.
                    add_special_tokens=True,      # Add '[CLS]' and '[SEP]'
                    max_length=self.max_seq_len,  # Pad & truncate.
                    pad_to_max_length=True,
                    return_attention_mask=True,   # Construct attn. masks.
                )
            )

            # Add the encoded sentence to the list.
            input_ids.append(encoded_dict['input_ids'])
            # And its attention mask (simply differentiates padding
            # from non-padding).
            attention_masks.append(encoded_dict['attention_mask'])

        return (
            np.array(input_ids, dtype=np.int32),
            np.array(attention_masks, dtype=np.int32)
        )

    def create_dataset(self, X, y, buffer_size=10000, train=True):

        input_ids, att_masks = self.tokenize_sentences(X)
        # Convert labels to int32
        labels = y

        def gen():
            for in_id, att_mk, label in zip(input_ids, att_masks, labels):
                yield ({"input_ids": in_id, "attention_mask": att_mk}, label,)

        return tf.data.Dataset.from_generator(
            gen, ({"input_ids": tf.int32,
                   "attention_mask": tf.int32}, tf.bool),
            ({"input_ids": tf.TensorShape([None]),
              "attention_mask": tf.TensorShape([None])}, tf.TensorShape([])),
        )

    def fit(self, X, y):
        X_train, X_val, y_train, y_val = (
            train_test_split(X, y, test_size=self.val_size)
        )

        print("Creating training set")
        ds_train = self.create_dataset(X_train, y_train)
        print("Creating validation set")
        ds_val = self.create_dataset(X_val, y_val)

        # %% colab={} colab_type="code" id="a-JK58wy8tPn"
        ds_train = (
            ds_train.shuffle(1000)
            .batch(self.batch_size)
            .repeat(self.n_epochs)
        )
        ds_val = ds_val.batch(self.batch_size)

        steps_per_epoch = X_train.shape[0] // self.batch_size
        validation_steps = X_val.shape[0] // self.batch_size

        # Prepare training: Compile tf.keras model with optimizer, loss and
        # learning rate schedule
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate,
                                             epsilon=1e-08, clipnorm=1.0)
        loss = tf.keras.losses.binary_crossentropy
        metric = tf.keras.metrics.binary_accuracy

        self.model.compile(optimizer=optimizer, loss=loss, metrics=[metric])

        # Parameters as defined in https://github.com/huggingface/transformers
        self.history = self.model.fit(ds_train, epochs=self.n_epochs,
                                      steps_per_epoch=steps_per_epoch,
                                      validation_data=ds_val,
                                      validation_steps=validation_steps)

