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

# %%
import torch
from torch.utils.data import (
    TensorDataset, DataLoader, 
    RandomSampler, SequentialSampler
)
from transformers import get_linear_schedule_with_warmup
from transformers import BertForSequenceClassification, AdamW, BertConfig
from transformers import BertTokenizer, BertModel, BertForMaskedLM
from sklearn.base import BaseEstimator, ClassifierMixin
from utils import get_reviews, results, MAX_LEN
from tqdm import tqdm
import numpy as np
import time
import datetime

BERTMODEL = "bert-base-uncased"

# %%
df = get_reviews()


# %% jupyter={"source_hidden": true}
class BertClaasifier(BaseEstimator, ClassifierMixin):
    
    def __init__(self, val_perc=0.1, max_len=MAX_LEN):
        # Load pre-trained model tokenizer (vocabulary)
        self.tokenizer = BertTokenizer.from_pretrained(BERTMODEL,
                                                       do_lower_case=True)
        self.max_len = MAX_LEN
        self.val_perc = val_perc
    
    def _tokenize(self, s):        
        # `encode_plus` will:
        #   (1) Tokenize the sentence.
        #   (2) Prepend the `[CLS]` token to the start.
        #   (3) Append the `[SEP]` token to the end.
        #   (4) Map tokens to their IDs.
        #   (5) Pad or truncate the sentence to `max_length`
        #   (6) Create attention masks for [PAD] tokens.
        encoded_dict = (
            self.tokenizer.encode_plus(
                s,                            # Sentence to encode.
                add_special_tokens = True,    # Add '[CLS]' and '[SEP]'
                max_length = self.max_len,    # Pad & truncate all sentences.
                pad_to_max_length = True,
                return_attention_mask = True, # Construct attn. masks.
                return_tensors = 'pt'         # Return pytorch tensors.
            )
        )
        
        # Return the encoded sentence and attention mask 
        return encoded_dict['input_ids'], encoded_dict['attention_mask']
    
    def _preprocess(self, X, y):
        # Tokenize all of the sentences and map the tokens to thier word IDs.
        input_ids = []
        attention_masks = []
        
        for s in tqdm(X):
            input_id, attention_mask = self._tokenize(s)
            # Add the encoded sentence to the list.   
            input_ids.append(input_id)
            # And its attention mask (simply differentiates padding from non-padding).
            attention_masks.append(attention_mask)
        
        # Convert the lists into tensors.
        input_ids = torch.cat(input_ids, dim=0)
        attention_masks = torch.cat(attention_masks, dim=0)
        labels = torch.tensor(y)
        
        # Combine the training inputs into a TensorDataset.
        return TensorDataset(input_ids, attention_masks, labels)
    
    @staticmethod
    def _get_model():
        # Load BertForSequenceClassification, the pretrained BERT model with a single 
        # linear classification layer on top. 
        return BertForSequenceClassification.from_pretrained(
            "bert-base-uncased", # Use the 12-layer BERT model, with an uncased vocab.
            num_labels = 2, # The number of output labels--2 for binary classification.
                            # You can increase this for multi-class tasks.   
            output_attentions = False, # Whether the model returns attentions weights.
            output_hidden_states = False, # Whether the model returns all hidden-states.
        )
    
    def _train(train_dataset, val_dataset):
        # Function to calculate the accuracy of our predictions vs labels
        def flat_accuracy(preds, labels):
            pred_flat = np.argmax(preds, axis=1).flatten()
            labels_flat = labels.flatten()
            return np.sum(pred_flat == labels_flat) / len(labels_flat)
        
        def format_time(elapsed):
            '''
            Takes a time in seconds and returns a string hh:mm:ss
            '''
            # Round to the nearest second.
            elapsed_rounded = int(round((elapsed)))

            # Format as hh:mm:ss
            return str(datetime.timedelta(seconds=elapsed_rounded))
        
        # The DataLoader needs to know our batch size for training, so we specify it 
        # here. For fine-tuning BERT on a specific task, the authors recommend a batch 
        # size of 16 or 32.
        batch_size = 32

        # Create the DataLoaders for our training and validation sets.
        # We'll take training samples in random order. 
        train_dataloader = DataLoader(
                    train_dataset,  # The training samples.
                    sampler = RandomSampler(train_dataset), # Select batches randomly
                    batch_size = batch_size # Trains with this batch size.
                )

        # For validation the order doesn't matter, so we'll just read them sequentially.
        validation_dataloader = DataLoader(
                    val_dataset, # The validation samples.
                    sampler = SequentialSampler(val_dataset), # Pull out batches sequentially.
                    batch_size = batch_size # Evaluate with this batch size.
        )
        
        model = BertClaasifier._get_model()

        # Tell pytorch to run this model on the GPU.
        model.cuda()
        
        # Note: AdamW is a class from the huggingface library (as opposed to pytorch) 
        # I believe the 'W' stands for 'Weight Decay fix"
        optimizer = AdamW(
            model.parameters(),
            lr = 2e-5, # args.learning_rate - default is 5e-5, our notebook had 2e-5
            eps = 1e-8 # args.adam_epsilon  - default is 1e-8.
        )

        # Number of training epochs. The BERT authors recommend between 2 and 4. 
        # We chose to run for 4, but we'll see later that this may be over-fitting the
        # training data.
        epochs = 4

        # Total number of training steps is [number of batches] x [number of epochs]. 
        # (Note that this is not the same as the number of training samples).
        total_steps = len(train_dataloader) * epochs

        # Create the learning rate scheduler.
        scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                    num_warmup_steps = 0, # Default value in run_glue.py
                                                    num_training_steps = total_steps)
    
        
    def fit(self, X, y):
        # Calculate the number of samples to include in each set.
        train_size = int(self.val_perc * len(dataset))
        val_size = len(dataset) - train_size
        
        dataset = _preprocess(X, y)

        # Divide the dataset by randomly selecting samples.
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        
        return self
    

# %%
df_ = df.head()

# %%
import tensorflow as tf
from transformers import *

# Load dataset, tokenizer, model from pretrained model/vocabulary
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
model = TFBertForSequenceClassification.from_pretrained('bert-base-cased')

# %%
# Prepare dataset for GLUE as a tf.data.Dataset instance
train_dataset = tokenizer(df_.review, tokenizer, max_length=128)

# %%
# Prepare dataset for GLUE as a tf.data.Dataset instance
train_dataset = glue_convert_examples_to_features(data['train'], tokenizer, max_length=128, task='mrpc')
valid_dataset = glue_convert_examples_to_features(data['validation'], tokenizer, max_length=128, task='mrpc')
train_dataset = train_dataset.shuffle(100).batch(32).repeat(2)
valid_dataset = valid_dataset.batch(64)

# Prepare training: Compile tf.keras model with optimizer, loss and learning rate schedule
optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5, epsilon=1e-08, clipnorm=1.0)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')
model.compile(optimizer=optimizer, loss=loss, metrics=[metric])

# Train and evaluate using tf.keras.Model.fit()
history = model.fit(train_dataset, epochs=2, steps_per_epoch=115,
                    validation_data=valid_dataset, validation_steps=7)

# Load the TensorFlow model in PyTorch for inspection
model.save_pretrained('./save/')
pytorch_model = BertForSequenceClassification.from_pretrained('./save/', from_tf=True)

# Quickly test a few predictions - MRPC is a paraphrasing task, let's see if our model learned the task
sentence_0 = "This research was consistent with his findings."
sentence_1 = "His findings were compatible with this research."
sentence_2 = "His findings were not compatible with this research."
inputs_1 = tokenizer.encode_plus(sentence_0, sentence_1, add_special_tokens=True, return_tensors='pt')
inputs_2 = tokenizer.encode_plus(sentence_0, sentence_2, add_special_tokens=True, return_tensors='pt')

pred_1 = pytorch_model(inputs_1['input_ids'], token_type_ids=inputs_1['token_type_ids'])[0].argmax().item()
pred_2 = pytorch_model(inputs_2['input_ids'], token_type_ids=inputs_2['token_type_ids'])[0].argmax().item()

print("sentence_1 is", "a paraphrase" if pred_1 else "not a paraphrase", "of sentence_0")
print("sentence_2 is", "a paraphrase" if pred_2 else "not a paraphrase", "of sentence_0")

# %%
r[2].shape

# %%
from transformers import pipeline

nlp = pipeline('feature-extraction')

# %%
type(nlp(df['review'][0]))

# %%
# Create a 90-10 train-validation split.

# Calculate the number of samples to include in each set.
train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size

# Divide the dataset by randomly selecting samples.
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

print('{:>5,} training samples'.format(train_size))
print('{:>5,} validation samples'.format(val_size))

# %%
# Print the original sentence.
print(' Original: ', df.loc[0, 'review'])

# Print the sentence split into tokens.
print('Tokenized: ', tokenizer.tokenize(df.loc[0, 'review']))

# Print the sentence mapped to token ids.
print('Token IDs: ', tokenizer.convert_tokens_to_ids(tokenizer.tokenize(df.loc[0, 'review'])))

# %%
# Tokenize input
tokenized_text = tokenizer.tokenize(df['review'])

# %%
# Mask a token that we will try to predict back with `BertForMaskedLM`
masked_index = 8
tokenized_text[masked_index] = '[MASK]'
assert tokenized_text == ['[CLS]', 'who', 'was', 'jim', 'henson', '?', '[SEP]', 'jim', '[MASK]', 'was', 'a', 'puppet', '##eer', '[SEP]']

# Convert token to vocabulary indices
indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
# Define sentence A and B indices associated to 1st and 2nd sentences (see paper)
segments_ids = [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]

# Convert inputs to PyTorch tensors
tokens_tensor = torch.tensor([indexed_tokens])
segments_tensors = torch.tensor([segments_ids])
