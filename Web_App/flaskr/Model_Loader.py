import nltk
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchtext
import spacy
import pickle
import dill
import unicodedata
import re
import os

from nltk import PorterStemmer
from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
from torchtext.legacy import data
from torch.utils.data import Dataset
from sklearn.base import BaseEstimator, TransformerMixin
from termcolor import colored
# from pycontractions import Contractions

class FilmClassifierLSTM(nn.Module):
    """ 
    Long-short term memory (LSTM) classifier
    Layers: Embedding -> LSTM -> fully connected
    
    Parameters:
        n_vocab: Number of words TEXT Field was trained on
        n_classes: Number of genres
        pad_index: Index of <pad> token
        unk_index: Index of <unk> token
        n_embedding: Size of the trained vectors, e.g if using 'glove.6B.100d', set to 100
        pretrained_embeddings: Vectors from pre-trained word embedding such as GloVe
        n_hidden: Number of hidden layers
        dropout: Dropout rate, e.g 0.2 = 20% dropout
        activation: Set as "softmax" or "sigmoid"
        bidirectiona: Whether to use bidirectional LSTM
        batch_norm: Whether to apply a batch normalization layer
    
    Return on forward pass:
        output: Predicted probabilities for each class
        
    """
    
    def __init__(self, n_vocab, n_classes, pad_index, unk_index, n_embedding, pretrained_embeddings=None,
                 n_hidden=256, dropout=0.2, activation="sigmoid", bidirectional=True, batch_norm=True):
        super().__init__()
        
        self.bidirectional = bidirectional
        self.batch_norm = batch_norm
        
        if bidirectional:
            # Use two layers for bidirectionality
            n_layers = 2
            # Double size of linear output
            linear_hidden = n_hidden * 2
        else:
            n_layers = 1
            linear_hidden = n_hidden
        
        # Create model layers
        self.dropout = nn.Dropout(dropout)
        self.embedding = nn.Embedding(n_vocab, n_embedding, padding_idx=pad_index) # Tell embedding not to learn <pad> embeddings
        self.lstm = nn.LSTM(n_embedding, n_hidden, num_layers=n_layers, dropout=dropout, bidirectional=bidirectional)
        self.batchnorm = nn.BatchNorm1d(linear_hidden)
        self.linear = nn.Linear(linear_hidden, n_classes)
        
        # Set output activation function
        if activation == "softmax":
            self.activation = nn.Softmax(dim=1)
        else:
            # Sigmoid recommended for multi-label
            self.activation = nn.Sigmoid()
        
        if pretrained_embeddings != None:
            # Replace weights of embedding layer
            self.embedding.weight.data.copy_(pretrained_embeddings)
            # Set padding and unknown tokens to zero
            self.embedding.weight.data[pad_index] = torch.zeros(n_embedding)
            self.embedding.weight.data[unk_index] = torch.zeros(n_embedding)

    def forward(self, text, text_lengths):
        # Create word embedding, then apply drop out
        embedded = self.embedding(text)
        dropped_embedded = self.dropout(embedded)
        # Pack the embedding so that LSTM only processes non-embedded sequences
        packed_embedded = nn.utils.rnn.pack_padded_sequence(dropped_embedded, text_lengths.to('cpu'))
        # Return output of all hidden states in the sequence, hidden state of the last LSTM unit and cell state
        packed_output, (hidden, cell) = self.lstm(packed_embedded)
        # Unpack packed_output
        unpacked_output, output_lengths = nn.utils.rnn.pad_packed_sequence(packed_output)
        
        if self.bidirectional:
            # Find the final two hidden states and join them together
            top_two_hidden = torch.cat((hidden[-1], hidden[-2]), dim=1)
            if self.batch_norm:
                top_two_hidden = self.batchnorm(top_two_hidden)
            # Apply dropout, pass through fully connected layer, then apply activation function
            output = self.activation(self.linear(self.dropout(top_two_hidden)))
        else:
            # Apply dropout to final hidden state, pass through fully connected layer, then apply activation function
            output = self.activation(self.linear(self.dropout(hidden[-1])))

        return output


class DescriptionTransformer(BaseEstimator, TransformerMixin):
    """
    Process the movie descriptions before classification
    
    Parameters:
        stop_words: The stop word list
        transformation: Lemmatization or stemming
        contractions: Set as True to contract words
        stemmer_algorithm: Algorithm to use when applying stemming, defaults to Porter if not given
        verbose: set as 0 to print nothing, 1 to print progress and 2 to print progress and data
    """
    
    def __init__(self, stop_words, transformation="lemmatize", contractions=False, 
                 stemmer_algorithm=None, verbose=0):
        # Settable parameters
        self.stop_words = stop_words
        self.transformation = transformation
        self.contractions = contractions
        self.stemmer_algorithm = stemmer_algorithm if stemmer_algorithm else PorterStemmer()
        self.verbose = verbose
        
        # Other
        self.data = None
        self.column_name = None
        
    def fit(self, x):
        if self.verbose > 0:
            print(colored("Called Description Transformer Fit", color="blue", attrs=['bold', 'underline']))
        return self
    
    def transform(self, x):
        if self.verbose > 0:
            print(colored("Called Description Transformer Transform", color="blue", attrs=['bold', 'underline']))
            print("Processing description text")
            
        # Copy the data and find the name of the description column
        self.data = x.copy()
        self.column_name = self.data.columns.values[0]
        
        # Load spaCy language processor
        nlp = spacy.load("en_core_web_sm")
        # Load pre-trained word embedding if using contractions
        # contraction = Contractions(api_key="glove-twitter-25") if self.contractions else None
        contraction = None
        # Process text by iterating over each sample's index and description
        for idx, sample in zip(self.data.index.values, self.data.values):
            # Change accented characters, e.g à -> a
            sample = self.remove_accents(str(sample))
            if contraction:
                None
                # Contract words, e.g "hasn't" -> "has not"
                # sample = list(contraction.expand_texts([sample], precise=True))
                # sample = ''.join(sample)

            # Input sample text into spaCy language processor
            doc = nlp(sample)
            # Split sample text into sentences
            sentences = list(doc.sents)
            
            for word_idx in range(len(sentences)):
                # Remove punctuation tokens, e.g. ! , .
                sentences[word_idx] = [token for token in sentences[word_idx] if not token.is_punct]
            
                # Remove stop words
                if self.stop_words:
                    sentences[word_idx] = [token for token in sentences[word_idx] if token.text.lower() not in self.stop_words]
            
                # Apply lemmatization
                if self.transformation[0].lower() == "l":
                    # Resolve words to their dictionary form using PoS tags
                    sentences[word_idx] = [token.lemma_.lower() for token in sentences[word_idx]]
                    
                # Apply stemming (only if lemmatization not applied)
                elif self.transformation[0].lower() == "s":
                    # Stem tokens
                    for char_idx in range(len(sentences[word_idx])):
                        # Apply stemmer to each word
                        stemmed = self.stemmer_algorithm.stem(sentences[word_idx][char_idx].text)
                        # Convert back to type Token and update word in sentence
                        sentences[word_idx][char_idx] = nlp(stemmed)[0]
                        
                # Remove remaining punctuation within tokens, e.g. "(years)" -> "years", not including -
                sentences[word_idx] = [token.translate(str.maketrans('', '', '!"#$%&\'()*+,./:;<=>?@[\\]^_`{|}~')) for token in sentences[word_idx]]
                
            # Split words containing dash or spaces caused by lemmatization, e.g. "16-year" -> "16" + "year"
            for k in range(len(sentences)):
                new_sentence = []
                for token in sentences[k]:
                    split_token = re.split(' |-', token)
                    for word in split_token:
                        # Check word not empty
                        if word:
                            new_sentence.append(word)
                # Replace words in sentence
                sentences[k] = new_sentence
                    
            # Remove empty lists from list of sentences
            sentences = [sent for sent in sentences if sent != []]
            # The join the sentences and update the descriptions dataframe
            word_list = [word for sent in sentences for word in sent]
            self.data.loc[idx, self.column_name] = ' '.join([str(elem) for elem in word_list])
            
#         if self.verbose > 1:
#             display(self.data)
        if self.verbose > 0:
            print(colored("Finshed processing all descriptions\n", color="blue", attrs=['bold', 'underline']))
            
        return self.data
    
    def remove_accents(self, text):
        # Remove accent or unknown characters from text
        text = unicodedata.normalize('NFD', text)               .encode('ascii', 'ignore')               .decode("utf-8")
        return str(text)


def text_to_genres(text, label_threshold=0.5, model_kwargs_file='model_kwargs.pickle',
                   model_weights_file='trained_model.pt', binary_encoder_file='binary_encoder.pickle',
                   TEXT_field_file="TEXT.Field", text_preprocessor_file="text_preprocessor.pickle"):

     # Load the text preprocessor transformer
    # text_preprocessor = pickle.load(open(text_preprocessor_file, 'rb'))
    text_preprocessor = DescriptionTransformer(stop_words=nltk.corpus.stopwords.words('english'), verbose=2)
    # Load the multi-hot binary encoder
    binary_encoder = pickle.load(open(binary_encoder_file, 'rb'))
    # Load TorchText TEXT field
    TEXT = dill.load(open(TEXT_field_file, "rb"))
    # Load the model parameters
    model_kwargs = pickle.load(open(model_kwargs_file, 'rb'))
    # Determine device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Convert text into dataframe to be compatible
    text_df = pd.DataFrame(data=[text], columns=["description"])
    # Process the text
    text_preprocessor.verbose = 0
    processed_text = text_preprocessor.transform(text_df)
    # Convert back to string
    processed_text = str(processed_text.values[0][0])

    # Get indexes of tokens
    token_indexes = [TEXT.vocab.stoi[token] for token in processed_text.split()]
    # Convert indexes to tensor
    token_tensor = torch.LongTensor(token_indexes).to(device)
    # Add extra dimension to shape to replicate batch
    token_tensor = token_tensor.unsqueeze(1)
    # Get the length of the text
    length_tensor = torch.LongTensor([len(token_indexes)])

    # Create the model
    model = FilmClassifierLSTM(**model_kwargs)
    # Set device
    model = model.to(device)
    # Load the model weights from file
    model.load_state_dict(torch.load(model_weights_file,map_location=torch.device('cpu')))
    # Set model to evaluation mode
    model.eval()

    # Make a prediction
    prediction = model(token_tensor, length_tensor)
    # Convert model outputs to binary labels, then to genre
    predicted_labels = torch.tensor([[1 if value > label_threshold else 0 for value in sample] for sample in prediction])
    if not 1 in predicted_labels:
        # Prevent no labels being predicted
        best_label = prediction.argmax(1)[0].item()
        predicted_labels[0][best_label] = 1

    # Calculate the percentage prediction
    predicted_categories_scores = []
    for idx in range(len(predicted_labels[0])):
        if predicted_labels[0][idx].item() == 1:
            predicted_categories_scores.append(prediction[0][idx].item())

    # Fit the encoder so it can be used
    binary_encoder.fit(binary_encoder.classes)
    # Convert the labels from binary to genres
    predicted_categories = binary_encoder.inverse_transform(predicted_labels.cpu())
    predicted_categories = list(predicted_categories[0])
    
    # put into dataframe
    data_3 = [(text, predicted_categories, predicted_categories_scores)]
    df = pd.DataFrame(data_3, columns=['text', 'genre', 'score'])
    
    return df

