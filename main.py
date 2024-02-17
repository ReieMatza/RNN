import os

import torch
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchdata.datapipes.iter import FileOpener
from torchtext import datasets
from torchtext.data import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torcheval.metrics.text import Perplexity
import torch.nn.functional as F


device = "cuda" if torch.cuda.is_available() else "cpu"
SEQUENCE_SIZE = 20
EMBEDDING_DIM = 6
HIDDEN_DIM = 200
BATCH_SIZE = 1
LEARNING_RATE = 0.001
MODEL_SAVE_PATH = r".\model.pt"
VOCAB_SAVE_PATH = r".\vocab.pt"
train_file = r'C:\Users\ReieMatza\PycharmProjects\RNN\dataset\ptb.train.txt'
valid_file = r'C:\Users\ReieMatza\PycharmProjects\RNN\dataset\ptb.valis.txt'
test_file = r"C:\Users\ReieMatza\PycharmProjects\RNN\dataset\ptb.test.txt"


class Ex2_dataset(torch.utils.data.Dataset):
    def __init__(self, path_to_file, vocab_path=None):
        self.path = path_to_file
        self.data, self.vocab = self.build_tokens_vocab(vocab_path)

    def build_tokens_vocab(self, vocab_path):
        tokenizer = get_tokenizer('basic_english')
        self.max_len = 0

        # Build the vocabulary
        def load_data(file_path):
            data = []
            with open(file_path, 'r', encoding='utf-8') as f:
                Lines = f.readlines()
                for l in Lines:
                    tokens = tokenizer(l)
                    self.max_len = len(tokens) if len(tokens) > self.max_len else self.max_len
                    data.append(tokens)
            return data

        data = load_data(self.path)
        if vocab_path is None:
            vocab = build_vocab_from_iterator(data, specials=["<eos>", "<unk>"])
            torch.save(vocab, VOCAB_SAVE_PATH)
        else:
            vocab = torch.load(VOCAB_SAVE_PATH)

        return data, vocab

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x = torch.as_tensor(self.vocab(self.data[index]))
        y = x[1:]
        x = F.pad(x, (0, self.max_len - len(x)), 'constant', self.vocab(["<eos>"])[0])
        y = F.pad(y, (0, self.max_len - len(y)), 'constant', self.vocab(["<eos>"])[0])
        return x,y


class LSTMReie(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int = EMBEDDING_DIM, hidden_dim: int = HIDDEN_DIM,
                 dropout: bool = False):
        super(LSTMReie, self).__init__()
        self.hidden_dim = hidden_dim
        self.word_embedding = nn.Embedding(vocab_size, EMBEDDING_DIM)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.fc = nn.Linear(self.hidden_dim, vocab_size)

    def forward(self, x, prev_state=None):
        embed = self.word_embedding(x)
        output, state = self.lstm(embed, prev_state) if prev_state else self.lstm(embed)
        logits = self.fc(output)

        return logits, state

    def init_state(self):
        return torch.zeros(BATCH_SIZE, self.hidden_dim), torch.zeros(BATCH_SIZE, self.hidden_dim)


def train(model: LSTMReie, train_data):
    train_loader = DataLoader(train_data, BATCH_SIZE)
    num_of_sequences = len(train_data)
    loss = nn.CrossEntropyLoss()
    model.train()
    preplexity = Perplexity()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    for i, (x, y) in enumerate(train_loader):
        x.to(device)
        y.to(device)
        if x.nelement() > 1:
            yi, (state_hi, state_ci) = model(x)
            cost = loss(torch.squeeze(yi),torch.squeeze(y))
            cost.backward()
            optimizer.step()

            # preplexity.update(yi,y)
            # prediction = torch.argmax(yi[-1])
            # preplexity{preplexity.compute()}
            if i % 1000 == 0:
                print(
                    f" sequence {i} out of {num_of_sequences}, {i * 100 / num_of_sequences}%")
    torch.save(model, MODEL_SAVE_PATH)


vocab_path = VOCAB_SAVE_PATH if os.path.isfile(VOCAB_SAVE_PATH) else None
train_data = Ex2_dataset(train_file, vocab_path)

train(LSTMReie(len(train_data.vocab)).to(device), train_data)
