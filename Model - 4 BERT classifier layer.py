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
#     name: python3
# ---

# %% id="xMjyof0kEsZp" colab_type="code" colab={}
# %%capture
# # !pip install transformers

# %% id="eu7iP06fGljZ" colab_type="code" colab={}
import tensorflow as tf
import tensorflow_datasets
from transformers import (
    TFBertForSequenceClassification, 
    TFDistilBertForSequenceClassification,
    DistilBertTokenizer, 
    BertTokenizer
)

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from utils import get_reviews, results, MAX_LEN
from tqdm import tqdm
import numpy as np
import pandas as pd
import time
import re
import datetime

BERTMODEL = "bert-base-uncased"
BATCH_SIZE = 32
N_EPOCHS = 2
MAX_LEN = 300


# %% id="y9j6AVfIISAa" colab_type="code" colab={}
def get_reviews():
    def preprocess(x):
        x = re.sub("<br\\s*/?>", " ", x)
        return x

    return (
        pd.read_csv("/content/drive/My Drive/Colab Notebooks/data/IMDB Dataset.csv")
        .assign(review=lambda df: df['review'].apply(preprocess))
    )


# %% id="vD0Q8b59FNDk" colab_type="code" colab={}
df = get_reviews()

# %% id="pr2agAwrFTtH" colab_type="code" colab={"base_uri": "https://localhost:8080/", "height": 51} outputId="02284ada-4abc-49ae-8c19-3fcb2b553243"
# %%time
# Load dataset, tokenizer, model from pretrained model/vocabulary
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-cased')
model = TFDistilBertForSequenceClassification.from_pretrained('distilbert-base-cased')

# %% id="JheWD0IMPipN" colab_type="code" colab={"base_uri": "https://localhost:8080/", "height": 85} outputId="b29dae0d-67fd-4ad4-d3d0-107300025715"
# Freezes all the layers (except the classifier and dropout)
for layer in model.layers:
    layer.trainable = False if layer.name == 'distilbert' else True
    print(layer.name, layer.trainable)


# %% id="9cyz5xQHFWzW" colab_type="code" colab={}
def tokenize_sentences(sentences, tokenizer, max_seq_len):
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
            tokenizer.encode_plus(
                sentence,                     # Sentence to encode.
                add_special_tokens = True,    # Add '[CLS]' and '[SEP]'
                max_length = MAX_LEN,         # Pad & truncate all sentences.
                pad_to_max_length = True,
                return_attention_mask = True, # Construct attn. masks.
            )
        )

        # Add the encoded sentence to the list.   
        input_ids.append(encoded_dict['input_ids'])
        # And its attention mask (simply differentiates padding from non-padding).
        attention_masks.append(encoded_dict['attention_mask'])
        
    return input_ids, attention_masks



def create_dataset(df, tokenizer, max_seq_len, 
                   epochs, batch_size, 
                   buffer_size=10000, train=True):
    
    input_ids, attention_masks = tokenize_sentences(df.review, tokenizer, max_seq_len)
    labels = 1 * (df.sentiment == 'positive')

    dataset = tf.data.Dataset.from_tensor_slices(((input_ids, attention_masks), labels.tolist()))
    if train:
        dataset = dataset.shuffle(buffer_size=buffer_size)
    dataset = dataset.repeat(epochs)
    dataset = dataset.batch(batch_size)
    if train:
        dataset = dataset.prefetch(1)
    
    return dataset


# %% id="mbPrwB2LI2wW" colab_type="code" colab={}
df_train, df_test, _, _ = train_test_split(df, df.sentiment, test_size=1/3)
df_train, df_val, _, _ = train_test_split(df_train, df_train.sentiment, test_size=0.1)

# %% id="0nabFdwrI4OR" colab_type="code" outputId="19433785-76f4-483b-fe4a-3ffa5326a7f1" colab={"base_uri": "https://localhost:8080/", "height": 34}
ds_train = create_dataset(df_train, tokenizer, max_seq_len=MAX_LEN, epochs=N_EPOCHS, batch_size=BATCH_SIZE)
ds_val = create_dataset(df_val, tokenizer, max_seq_len=MAX_LEN, epochs=N_EPOCHS, batch_size=BATCH_SIZE)

# %% id="V-ns-jOWI5eq" colab_type="code" colab={}
steps_per_epoch = df_train.shape[0] // BATCH_SIZE
validation_steps = df_val.shape[0] // BATCH_SIZE

# Prepare training: Compile tf.keras model with optimizer, loss and learning rate schedule
optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5, epsilon=1e-08, clipnorm=1.0)
# loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
# metric = tf.keras.metrics.SparseCategoricalCrossentropy('accuracy')
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train and evaluate using tf.keras.Model.fit()
# history = model.fit(ds_train, epochs=NR_EPOCHS, steps_per_epoch=steps_per_epoch, 
#                     validation_data=ds_val, validation_steps=validation_steps)

# Parameters as defined in https://github.com/huggingface/transformers
history = model.fit(ds_train, epochs=N_EPOCHS, steps_per_epoch=115,
                    validation_data=ds_val, validation_steps=7)

# %% id="qxcRpAQ5R_6F" colab_type="code" colab={}
