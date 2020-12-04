import random

import spacy
import torch
import torch.nn as nn
import torch.optim as optim
from torchtext import data
import sys
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

nlp = spacy.load('en')
SEED = 1731

torch.manual_seed(SEED)

torch.backends.cudnn.deterministic = True

medications = {}
df = pd.read_csv('dataset/medication_names.csv', header=None)
for row in df.values:
    drugs = list(row)
    drugs = [x for x in drugs if pd.notnull(x)]
    size = len(drugs)
    key = drugs[0]
    drugs.pop(0)
    medications[key] = drugs
print(len(medications))

TEXT = data.Field(tokenize='spacy', batch_first=True, include_lengths=True)
LABEL = data.LabelField(dtype=torch.float, batch_first=True)

fields = [(None, None), (None, None), ('text', TEXT), ('label', LABEL)]
path = sys.argv[6]
print('path', path)
N_EPOCHS = int(sys.argv[4])
print('N_EPOCHS', N_EPOCHS)
BATCH_SIZE = int(sys.argv[2])
print('BATCH_SIZE', BATCH_SIZE)
output_path = sys.argv[8]

training_data = data.TabularDataset(path=path, format='csv', fields=fields, skip_header=True)

train_data, validation_data = training_data.split(split_ratio=0.75, random_state=random.seed(SEED))
print(type(train_data))

TEXT.build_vocab(train_data, min_freq=3, vectors='glove.twitter.27B.100d')
LABEL.build_vocab(train_data)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_iterator, valid_iterator = data.BucketIterator.splits(
    (train_data, validation_data),
    batch_size=BATCH_SIZE,
    sort_key=lambda x: len(x.text),
    sort_within_batch=True,
    device=device)


class classifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, bidirectional, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers, bidirectional=bidirectional,
                            dropout=dropout, batch_first=True)
        self.bn = nn.BatchNorm1d(hidden_dim * 2)
        self.relu1 = nn.RReLU()
        self.relu2 = nn.RReLU()
        self.fc1 = nn.Linear(hidden_dim * 2, output_dim)
        self.act1 = nn.Sigmoid()
        self.act2 = nn.Sigmoid()

    def forward(self, text, text_lengths):
        embedded = self.embedding(text)
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths, batch_first=True)
        packed_output, (x, cell) = self.lstm(packed_embedded)
        x = torch.cat((x[-2, :, :], x[-1, :, :]), dim=1)
        x = self.bn(x)
        x = self.relu1(x)
        x = self.relu2(x)
        x = self.act1(x)
        x = self.fc1(x)
        x = self.act2(x)
        return x


size_of_vocab = len(TEXT.vocab)
embedding_dim = 100
num_hidden_nodes = 64
num_output_nodes = 1
num_layers = 4
bidirection = True
dropout = 0.2
model = classifier(size_of_vocab, embedding_dim, num_hidden_nodes, num_output_nodes, num_layers, bidirectional=True,
                   dropout=dropout)
print(model)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


print(f'The model has {count_parameters(model):,} trainable parameters')
pretrained_embeddings = TEXT.vocab.vectors
model.embedding.weight.data.copy_(pretrained_embeddings)

print(pretrained_embeddings.shape)

learning_rate = 1e-3
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.BCELoss()


# define metric
def binary_accuracy(preds, y):
    rounded_preds = torch.round(preds)
    correct = (rounded_preds == y).float()
    acc = correct.sum() / len(correct)
    return acc


model = model.to(device)
criterion = criterion.to(device)


def train(model, iterator, optimizer, criterion):
    epoch_loss = 0
    epoch_acc = 0

    model.train()

    for batch in iterator:
        optimizer.zero_grad()
        text, text_lengths = batch.text
        predictions = model(text, text_lengths).squeeze(1)
        loss = criterion(predictions, batch.label)
        acc = binary_accuracy(predictions, batch.label)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        epoch_acc += acc.item()
    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def evaluate(model, iterator, criterion):
    epoch_loss = 0
    epoch_acc = 0
    model.eval()
    with torch.no_grad():
        for batch in iterator:
            text, text_lengths = batch.text
            predictions = model(text, text_lengths).squeeze()
            loss = criterion(predictions, batch.label)
            acc = binary_accuracy(predictions, batch.label)
            epoch_loss += loss.item()
            epoch_acc += acc.item()
    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def predict(model, sentence):
    tokenized = [tok.text for tok in nlp.tokenizer(sentence)]  # tokenize the sentence
    indexed = [TEXT.vocab.stoi[t] for t in tokenized]  # convert to integer sequence
    length = [len(indexed)]  # compute no. of words
    tensor = torch.LongTensor(indexed).to(device)  # convert to tensor
    tensor = tensor.unsqueeze(1).T  # reshape in form of batch,no. of words
    length_tensor = torch.LongTensor(length)  # convert to tensor
    prediction = model(tensor, length_tensor).squeeze()  # prediction
    return prediction.item()


y_test = []
x_test = []
for row in validation_data:
    label = int(row.label)
    y_test.append(label)
    text = ' '.join(map(str, row.text))
    x_test.append(text)

output_csv_rows = []
output_csv_rows.append(['Tweet', 'Has_medication', 'Begin', 'End', 'Span', 'Drug normalized'])


def check_med(text):
    flag = True
    for item in medications:
        for drug in medications[item]:
            if drug in text:
                print(text, ",", drug, ",", text.find(drug))
                flag = False
                output_csv_rows.append([text, 1, int(text.find(drug)), int(text.find(drug) + len(drug)), drug, item])
                break
    if flag:
        output_csv_rows.append([text, 1])


def calc_accuracy():
    y_pred = []
    idx = 0
    for row in x_test:
        prediction = predict(model, row)
        if prediction >= 0.5:
            y_pred.append(1)
            # print(row)
            check_med(row)
        else:
            y_pred.append(0)
            out_data = [row, 0]
            output_csv_rows.append(out_data)
        idx = idx + 1
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))


best_valid_loss = float('inf')
for epoch in range(N_EPOCHS):
    train_loss, train_acc = train(model, train_iterator, optimizer, criterion)
    valid_loss, valid_acc = evaluate(model, valid_iterator, criterion)
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
    print("epoch ", epoch)
    print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc * 100:.2f}%')

calc_accuracy()
df = pd.DataFrame(output_csv_rows)
df.to_csv(output_path, index=False, header=False)
