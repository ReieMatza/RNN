import torch
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchdata.datapipes.iter import FileOpener
from torchtext import datasets
from torchtext.data import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torcheval.metrics.text import Perplexity

device = "cuda" if torch.cuda.is_available() else "cpu"
EMBEDDING_DIM = 6
HIDDEN_DIM=200
BATCH_SIZE = 1
LEARNING_RATE = 0.001
SAVE_PATH = r".\model.pt"
train_file = r'C:\Users\ReieMatza\PycharmProjects\RNN\dataset\ptb.train.txt'
valid_file = r'C:\Users\ReieMatza\PycharmProjects\RNN\dataset\ptb.valis.txt'
test_file = r"C:\Users\ReieMatza\PycharmProjects\RNN\dataset\ptb.test.txt"


class Ex2_dataset(torch.utils.data.Dataset):
    def __init__(self, path_to_file):
        self.path = path_to_file
        self.data, self.vocab = self.build_tokens_vocab()

    def build_tokens_vocab(self):
        tokenizer = get_tokenizer('basic_english')

        # Build the vocabulary
        def build_vocab(file_path):
            data = []
            with open(file_path, 'r', encoding='utf-8') as f:
                Lines = f.readlines()
                for l in Lines:
                    data.append(tokenizer(l))

            return data

        train_tokens = build_vocab(self.path)
        vocab = build_vocab_from_iterator(train_tokens, specials=["<unk>"])
        return train_tokens, vocab

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return torch.as_tensor(self.vocab(self.data[index][:-1])), torch.as_tensor(self.vocab([self.data[index][-1]]))


class LSTMReie(nn.Module):
    def __init__(self, vocab_size: int , embedding_dim: int = EMBEDDING_DIM, hidden_dim: int = HIDDEN_DIM , dropout: bool = False):
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
        return torch.zeros( BATCH_SIZE, self.hidden_dim), torch.zeros(BATCH_SIZE, self.hidden_dim)


def train(model: LSTMReie, train_data):
    train_loader = DataLoader(train_data, BATCH_SIZE)
    num_of_sequences = len(train_data)
    model.train()
    loss_fun = Perplexity()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    for i, (x, y) in enumerate(train_loader):
        if x.nelement()>1:
            state_hi, state_ci = model.init_state()
            yi, (state_hi, state_ci) = model(torch.squeeze(x), (state_hi, state_ci))

            loss = loss_fun.update(torch.unsqueeze(yi,0),x)
            prediction = torch.argmax(yi[-1])
            loss.compute()
            optimizer.step()
            if i % 100 == 0:
                print(f" sequence {i} out of {num_of_sequences}, {i*100/num_of_sequences}%")
    torch.save(model, SAVE_PATH)


train_data = Ex2_dataset(train_file)

train(LSTMReie(len(train_data.vocab)), train_data)