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
    DistilBertConfig
)
from transformers.tokenization_distilbert import DistilBertTokenizerFast

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split
import tensorflow as tf

from tqdm import tqdm
import numpy as np
import os
from utils import MAX_LEN


BERTMODEL = "distilbert-base-cased"
MODEL_PATH = './model'
BATCH_SIZE = 32
N_EPOCHS = 30
LEARNING_RATE = 3e-5


class BertClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, max_seq_len=MAX_LEN, batch_size=BATCH_SIZE,
                 n_epochs=N_EPOCHS, val_size=0.1,
                 learning_rate=LEARNING_RATE,
                 load_local_pretrained=False):

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

        if load_local_pretrained:
            self.model = (
                TFDistilBertForSequenceClassification
                .from_pretrained(MODEL_PATH)
            )

        else:
            config = DistilBertConfig.from_pretrained(BERTMODEL, num_labels=2)
            self.model = (
                TFDistilBertForSequenceClassification
                .from_pretrained(BERTMODEL, config=config)
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

    def create_dataset(self, X, y, train=True):
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

        ds_train = (
            self.create_dataset(X_train.tolist(), y_train)
            .shuffle(1000).batch(self.batch_size).repeat(-1)
        )
        ds_val = (
            self.create_dataset(X_val.tolist(), y_val)
            .batch(self.batch_size * 2).repeat(-1)
        )

        # Prepare training: Compile tf.keras model with optimizer, loss
        # and learning rate schedule
        opt = tf.keras.optimizers.Adam(learning_rate=self.learning_rate,
                                       epsilon=1e-08)

        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        metric = tf.keras.metrics.SparseCategoricalAccuracy("accuracy")

        self.model.compile(optimizer=opt, loss=loss, metrics=[metric])

        # Train and evaluate using tf.keras.Model.fit()
        train_steps = X_train.shape[0] // self.batch_size
        val_steps = X_val.shape[0] // self.batch_size

        self.history = self.model.fit(
            ds_train,
            epochs=self.n_epochs,
            steps_per_epoch=train_steps,
            validation_data=ds_val,
            validation_steps=val_steps,
        )

    def save_model(self):
        if self.fit:
            # Save TF2 model
            os.makedirs(MODEL_PATH, exist_ok=True)
            self.model.save_pretrained(MODEL_PATH)
        else:
            print("The model is not trained")

    def predict(self, X):
        n_batches = X.shape[0] // self.batch_size
        input_ids, _ = self.tokenize_sentences(X.tolist())
        y_pred = []
        input_ids_batches = np.array_split(input_ids, n_batches)
        for batch in tqdm(input_ids_batches):
            y_pred.append(self.model(batch)[0].numpy().argmax(axis=1))
        return np.concatenate(y_pred)
