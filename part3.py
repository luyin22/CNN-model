import numpy as np
import torch
import torch.nn as tnn
import torch.nn.functional as F
import torch.optim as topti
from torchtext import data
from torchtext.vocab import GloVe
from imdb_dataloader import IMDB
import re


# Class for creating the neural network.
class Network(tnn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.gru = torch.nn.GRU(50, 200, num_layers=2, batch_first=True, dropout=0.7)
        self.ln1 = torch.nn.Linear(200, 128)
        self.rl = torch.nn.ReLU()
        self.ln2 = torch.nn.Linear(128, 1)

    def forward(self, input, length):
        """
        DO NOT MODIFY FUNCTION SIGNATURE
        Create the forward pass through the network.
        """
        x, _ = self.gru(input)
        x = self.ln1(x[np.arange(input.shape[0]), length-1])
        x = self.rl(x)
        x = self.ln2(x).squeeze()
        return x

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, torch.nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, torch.nn.Linear):
                torch.nn.init.normal_(m.weight.data, 0, 0.01)
                m.bias.data.zero_()


class PreProcessing():
    def pre(x):
        """Called after tokenization"""
        long_x = ' '.join(x)
        long_x = re.sub('<br />', ' ', long_x)
        long_x = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", long_x)
        long_x = re.sub(r"\'s", " is", long_x)
        long_x = re.sub(r"\'ve", " have", long_x)
        long_x = re.sub(r"n\'t", " not", long_x)
        long_x = re.sub(r"\'re", " are", long_x)
        long_x = re.sub(r"\'d", " would", long_x)
        long_x = re.sub(r"\'ll", " will", long_x)
        long_x = re.sub(r'won\'t', 'will not', long_x)
        long_x = re.sub(r'can\'t', 'can not', long_x)
        long_x = re.sub(r'\'m', ' am', long_x)
        long_x = re.sub(r'\'t', ' not', long_x)
        long_x = re.sub(r",", " , ", long_x)
        long_x = re.sub(r"!", " ! ", long_x)
        long_x = re.sub(r"\(", " \( ", long_x)
        long_x = re.sub(r"\)", " \) ", long_x)
        long_x = re.sub(r"\?", " \? ", long_x)
        long_x = re.sub(r"\s{2,}", " ", long_x)
        x = long_x.split(' ')
        return x

    def post(batch, vocab):
        """Called after numericalization but prior to vectorization"""
        return batch
    
    stopwords = ["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your",
                "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "her",
                "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves",
                "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was",
                "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing",
                "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by",
                "for", "with", "about", "against", "between", "into", "through", "during", "before", "after",
                "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again",
                "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both",
                "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same",
                "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"]
    text_field = data.Field(lower=True, include_lengths=True, batch_first=True, preprocessing=pre, postprocessing=post, stop_words=stopwords)


def lossFunc():
    """
    Define a loss function appropriate for the above networks that will
    add a sigmoid to the output and calculate the binary cross-entropy.
    """
    return torch.nn.BCEWithLogitsLoss()

def main():
    # Use a GPU if available, as it should be faster.
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using device: " + str(device))

    # Load the training dataset, and create a data loader to generate a batch.
    textField = PreProcessing.text_field
    labelField = data.Field(sequential=False)

    train, dev = IMDB.splits(textField, labelField, train="train", validation="dev")

    textField.build_vocab(train, dev, vectors=GloVe(name="6B", dim=50))
    labelField.build_vocab(train, dev)

    trainLoader, testLoader = data.BucketIterator.splits((train, dev), shuffle=True, batch_size=64,
                                                         sort_key=lambda x: len(x.text), sort_within_batch=True)

    net = Network().to(device)
    net.initialize_weights()
    criterion =lossFunc()
    optimiser = topti.Adam(net.parameters(), lr=0.001)  # Minimise the loss using the Adam algorithm.

    for epoch in range(10):
        running_loss = 0

        for i, batch in enumerate(trainLoader):
            # Get a batch and potentially send it to GPU memory.
            inputs, length, labels = textField.vocab.vectors[batch.text[0]].to(device), batch.text[1].to(
                device), batch.label.type(torch.FloatTensor).to(device)

            labels -= 1

            # PyTorch calculates gradients by accumulating contributions to them (useful for
            # RNNs).  Hence we must manually set them to zero before calculating them.
            optimiser.zero_grad()

            # Forward pass through the network.
            output = net(inputs, length)

            loss = criterion(output, labels)

            # Calculate gradients.
            loss.backward()

            # Minimise the loss according to the gradient.
            optimiser.step()

            running_loss += loss.item()

            if i % 32 == 31:
                print("Epoch: %2d, Batch: %4d, Loss: %.3f" % (epoch + 1, i + 1, running_loss / 32))
                running_loss = 0

    num_correct = 0

    # Save mode
    torch.save(net.state_dict(), "./model.pth")
    print("Saved model")

    # Evaluate network on the test dataset.  We aren't calculating gradients, so disable autograd to speed up
    # computations and reduce memory usage.
    with torch.no_grad():
        for batch in testLoader:
            # Get a batch and potentially send it to GPU memory.
            inputs, length, labels = textField.vocab.vectors[batch.text[0]].to(device), batch.text[1].to(
                device), batch.label.type(torch.FloatTensor).to(device)

            labels -= 1

            # Get predictions
            outputs = torch.sigmoid(net(inputs, length))
            predicted = torch.round(outputs)

            num_correct += torch.sum(labels == predicted).item()

    accuracy = 100 * num_correct / len(dev)

    print(f"Classification accuracy: {accuracy}")

if __name__ == '__main__':
    main()
