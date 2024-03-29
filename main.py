import os
from typing import Union

import torch
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchtext.data import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torcheval.metrics.text import Perplexity
import torch.nn.functional as F
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"
SEQUENCE_LEN = 20
EMBEDDING_DIM = 200
NUM_OF_EPOCHS = 15
HIDDEN_DIM = 200
MINI_BATCH_SIZE = 20
LEARNING_RATE = 0.001
NUM_LAYERS = 2
MODEL = "LSTM"
DROPOUT = 0.0
MODEL_SAVE_PATH = rf"{MODEL}_{DROPOUT}.pt"
VOCAB_SAVE_PATH = r".\vocab.pt"
PLOT_TITLE = f"preplexity vs epoc for {MODEL} Dropout = {DROPOUT}"
train_file = os.path.join(os.getcwd(), "dataset", 'ptb.train.txt')
valid_file = os.path.join(os.getcwd(), "dataset", 'ptb.valid.txt')
test_file = os.path.join(os.getcwd(), "dataset", 'ptb.test.txt')


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
        x = torch.as_tensor(self.vocab(self.data[index][0:SEQUENCE_LEN]))
        y = x[1:]
        x = F.pad(x, (0, SEQUENCE_LEN - len(x)), 'constant', self.vocab(["<eos>"])[0])
        y = F.pad(y, (0, SEQUENCE_LEN - len(y)), 'constant', self.vocab(["<eos>"])[0])
        return x, y


class MYLSTM(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int = EMBEDDING_DIM, hidden_dim: int = HIDDEN_DIM, dropout=0.0,
                 num_layers=1):
        super(MYLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.word_embedding = nn.Embedding(vocab_size, EMBEDDING_DIM)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, dropout=dropout, num_layers=num_layers)
        self.fc = nn.Linear(self.hidden_dim, vocab_size)

        self.word_embedding.weight.data.uniform_(-0.1, 0.1)
        self.fc.weight.data.uniform_(-0.1, 0.1)

    def forward(self, x, prev_state=None):
        embed = self.word_embedding(x)
        output, state = self.lstm(embed, prev_state) if prev_state else self.lstm(embed)
        logits = self.fc(output)

        return logits, state

    def get_init_state(self):
        state_hi = torch.zeros((NUM_LAYERS, SEQUENCE_LEN, HIDDEN_DIM)).to(device)
        state_ci = torch.zeros((NUM_LAYERS, SEQUENCE_LEN, HIDDEN_DIM)).to(device)
        return state_hi, state_ci


class MYGRU(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int = EMBEDDING_DIM, hidden_dim: int = HIDDEN_DIM, dropout=0.0,
                 num_layers=1):
        super(MYGRU, self).__init__()
        self.hidden_dim = hidden_dim
        self.word_embedding = nn.Embedding(vocab_size, EMBEDDING_DIM)
        self.gru = nn.GRU(embedding_dim, hidden_dim, dropout=dropout, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(self.hidden_dim, vocab_size)

        self.word_embedding.weight.data.uniform_(-0.1, 0.1)
        self.fc.weight.data.uniform_(-0.1, 0.1)

    def forward(self, x, prev_state=None):
        embed = self.word_embedding(x)
        output, state = self.gru(embed, prev_state) if prev_state is not None else self.gru(embed)
        logits = self.fc(output)

        return logits, state

    def get_init_state(self):
        state_hi = torch.zeros((NUM_LAYERS, MINI_BATCH_SIZE, HIDDEN_DIM)).to(device)
        return state_hi


def prep_plot(training_prep_list, testing_prep_list):
    plt.plot(range(1, len(testing_prep_list) + 1), testing_prep_list, label='test data')
    plt.plot(range(1, len(training_prep_list) + 1), training_prep_list, label='train data')
    plt.title(PLOT_TITLE)
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("acc %")
    plt.show()


def preplexity(model: nn.Module, dataset):
    model.eval()
    dataloader = DataLoader(dataset, MINI_BATCH_SIZE)
    preplexity = Perplexity().to(device)
    with torch.no_grad():
        for batch, (x, y) in enumerate(dataloader):
            if x.size(0) == MINI_BATCH_SIZE:
                x = x.to(device)
                y = y.to(device)
                yi, _ = model(x)
                preplexity.update(yi, y)
    return preplexity.compute().cpu()


def train(model: MYLSTM, train_data, test_data):
    training_prep, testing_prep = [], []
    num_of_mini_batches = len(train_data) / MINI_BATCH_SIZE

    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
    train_loader = DataLoader(train_data, MINI_BATCH_SIZE)

    # if MODEL == "LSTM":
    #     state_hi, _ = model.get_init_state()
    # elif MODEL == "GRU":
    #     state_hi = model.get_init_state()

    for epoch in range(NUM_OF_EPOCHS):
        model.train()
        for batch, (x, y) in enumerate(train_loader):
            # if x.size(0) == MINI_BATCH_SIZE:
            optimizer.zero_grad()
            # if MODEL == "LSTM":
            #     state_hi = state_hi.detach()
            #     # state_ci = state_ci.detach()
            # elif MODEL == "GRU":
            #     state_hi = state_hi.detach()

            x = x.to(device)
            y = y.to(device)

            # Forward pass
            # if MODEL == "LSTM":
            #     _, state_ci = model.get_init_state()
            #     yi, (state_hi, _) = model(x, (state_hi, state_ci))
            #     yi, _ = model(x)
            # elif MODEL == "GRU":
            yi, _ = model(x)

            yi = yi.reshape(-1, len(train_data.vocab))
            y = y.reshape(-1)
            cost = loss(yi, y)

            # Backward and optimize
            cost.backward()
            optimizer.step()
        if epoch >= 5:
            scheduler.step()

        preplexity_train = preplexity(model, train_data)
        training_prep.append(preplexity_train)

        preplexity_test = preplexity(model, test_data)
        testing_prep.append(preplexity_test)

        print(
            f"Epoch {epoch + 1} ot of {NUM_OF_EPOCHS}, preplexity_train {preplexity_train}, preplexity_test {preplexity_test}")
    torch.save(model, MODEL_SAVE_PATH)
    return training_prep, testing_prep


def play(sequence: str):
    model = torch.load(MODEL_SAVE_PATH).to(device)
    vocab = torch.load(VOCAB_SAVE_PATH)
    input = sequence.split()
    input = torch.as_tensor(vocab(input)).to(device)
    yi,_ = model(input)
    words = vocab.lookup_tokens(torch.argmax(yi,1).tolist())
    print(f"input: {sequence}")
    print(f"output: {words}")


def main():
    vocab_path = VOCAB_SAVE_PATH if os.path.isfile(VOCAB_SAVE_PATH) else None
    train_data = Ex2_dataset(train_file, vocab_path)
    test_data = Ex2_dataset(test_file, vocab_path)
    vocab_size = len(train_data.vocab)
    if MODEL == "LSTM":
        training_prep, testing_prep = train(MYLSTM(vocab_size, dropout=DROPOUT, num_layers=NUM_LAYERS).to(device),
                                            train_data, test_data)
    else:
        training_prep, testing_prep = train(MYGRU(vocab_size, dropout=DROPOUT, num_layers=NUM_LAYERS).to(device),
                                            train_data, test_data)

    prep_plot(training_prep, testing_prep)


def final_table():
    vocab_path = VOCAB_SAVE_PATH if os.path.isfile(VOCAB_SAVE_PATH) else None
    train_data = Ex2_dataset(train_file, vocab_path)
    test_data = Ex2_dataset(test_file, vocab_path)
    valid_data = Ex2_dataset(valid_file, vocab_path)
    models_path = ["LSTM_0.0.pt","LSTM_0.5.pt","GRU_0.0.pt","GRU_0.5.pt"]
    for i_m in models_path:
        model = torch.load(i_m)
        train_prep=preplexity(model,train_data)
        test_prep = preplexity(model, test_data)
        valid_prep = preplexity(model, valid_data)
        print(f"model {i_m}: train_prep {train_prep}, test_prep {test_prep}, valid_prep {valid_prep}")



# main()
final_table()
# play("although preliminary findings were reported more than a year ago the latest results appear in today")


