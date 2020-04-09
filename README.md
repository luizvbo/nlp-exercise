# nlp-exercise

## Exercise 

Make a plot of accuracy as a function of the number of data points in the training set for the models below. 

Models to compare:
* Make a baseline model (simple model, from e.g. sklearn)
* Make an RNN based model and fit the word embeddings yourself
* Make an RNN based model, but use word embeddings from word2vec or document tensor from Spacy (see hint below) as input (i.e. do not start with an embedding layer!)
* Use a pretrained BertModel from HuggingFace, only fit the classifier layers
* Fit a sklearn model on the pooled output of a pretrained BERT model

Compute intensive jobs (Bonus):
* Fine-tune a pretrained BertModel (i.e. fit all layers, might need a GPU for this)
* Re-initialize the weights of the BERT model and fit from scratch 
