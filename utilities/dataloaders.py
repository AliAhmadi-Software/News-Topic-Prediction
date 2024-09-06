import torch

import numpy as np

from torch import tensor
from ast import literal_eval

from torch.nn.utils.rnn import pad_sequence
from torchtext.data.utils import get_tokenizer
from torch.utils.data import Dataset, DataLoader

from utilities.utils import dict2w


class Vocabulary:
    ''' the vocabulary class for news classification on the 'stories.csv' dataset '''
    def __init__(self, freq_thresh=5):
        # initializing parameters
        self.freq_thresh = freq_thresh

        # setting initial tokens
        self.InS = dict2w()
        self.InS[0] = '<PAD>'
        self.InS[1] = '<SOS>'
        self.InS[2] = '<EOS>'
        self.InS[3] = '<UNK>'

        # initializing the tokenizer
        self.tokenizer = get_tokenizer('basic_english', language='en')

    def __len__(self):
        return len(self.InS)

    def tokenizer_eng(self, text):
        return self.tokenizer(text)

    def build_vocabulary(self, sentence_list):
        # building a vocabulary of the dataset tokens
        frequencies = {}
        idx = len(self.InS)
        for sentence in sentence_list:
            for word in self.tokenizer_eng(sentence):
                if word not in frequencies:
                    frequencies[word] = 1
                else:
                    frequencies[word] += 1
                if frequencies[word] == self.freq_thresh:
                    self.InS[word] = idx
                    idx += 1

    def numericalize(self, text):
        # stoi using the vocabulary
        tokenized_text = self.tokenizer_eng(text)

        return [
            self.InS[token] if token in self.InS else self.InS['<UNK>']
            for token in tokenized_text
        ]


class news_dataset(Dataset):
    ''' a dataset class for the 'stories.csv' dataset '''
    def __init__(self, data, freq_thresh=5, vocabulary=None, inference=True):
        self.inference = inference

        self.body = data['body']
        if not self.inference:
            self.topic = data['topic']

        # initializing vocabulary in case of having no vocab argument
        if not vocabulary:
            self.vocab = Vocabulary(freq_thresh)
            self.vocab.build_vocabulary(self.body.tolist())
        else: self.vocab = vocabulary

    def __len__(self):
        return len(self.body)

    def __getitem__(self, idx):
        # processing the text data
        text = self.body.iloc[idx]
        numerical_text = [self.vocab.InS['<SOS>']]
        numerical_text += self.vocab.numericalize(text)
        numerical_text.append(self.vocab.InS['<EOS>'])

        if not self.inference:
            # creating a tensor of labels
            label = literal_eval(self.topic.iloc[idx])
            label = tensor(label, dtype=torch.int64)
            return tensor(numerical_text, dtype=torch.int64), label
        else: return tensor(numerical_text, dtype=torch.int64)


class news_collate:
    ''' this class concats records in batches for training '''
    def __init__(self, pad_idx, inference=False):
        self.pad_idx = pad_idx
        self.inference = inference

    def __call__(self, batch):
        if not self.inference:
            # creating a padded tensor of numerical text representations
            texts = [item[0] for item in batch]
            texts = pad_sequence(texts, batch_first=True,
                                padding_value=self.pad_idx)
            
            # creating a tensor of all labels
            labels = torch.cat([item[1].unsqueeze(0)
                            for item in batch], dim=0).type(torch.float)
            return texts, labels
        else:
            # creating a padded tensor of numerical text representations
            texts = pad_sequence(batch, batch_first=True,
                                padding_value=self.pad_idx)
            return texts


def news_loader(data, batch_size=32, shuffle=True, vocab=None, inference=False):
    ''' this function returns dataloaders for training the news classifier '''
    # initializing the vocabulary and the padding value
    if not vocab:
        vocab = Vocabulary()
        vocab.build_vocabulary(data['body'].tolist())
        pad_idx = vocab.InS['<PAD>']
    else: 
        pad_idx = vocab.InS['<PAD>']

    if not inference:
        # splitting train and validation sets
        mask = np.random.rand(len(data)) < .9
        train_data = data[mask]
        validation_data = data[~mask]

        # creating training and validation datasets
        train_dataset = news_dataset(train_data, vocabulary=vocab)
        val_dataset = news_dataset(validation_data, vocabulary=vocab)

        # creating training and validation dataloaders
        train_loader = DataLoader(dataset=train_dataset,
                                batch_size=batch_size,
                                shuffle=shuffle,
                                collate_fn=news_collate(pad_idx=pad_idx, inference=inference))

        val_loader = DataLoader(dataset=val_dataset,
                                batch_size=batch_size,
                                shuffle=shuffle,
                                collate_fn=news_collate(pad_idx=pad_idx, inference=inference))

        return vocab, train_loader, val_loader
    else:
        dataset = news_dataset(data, vocabulary=vocab)
        dataloader = DataLoader(dataset=dataset,
                                batch_size=batch_size,
                                shuffle=shuffle,
                                collate_fn=news_collate(pad_idx=pad_idx, inference=inference))
        return dataloader


