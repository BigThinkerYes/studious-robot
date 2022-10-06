import torch
import torch.nn as nn


class NeuralNet(nn.Module):
    # feed-forward neural net with two hidden layers
    # gets bag of words as an input
    # one layer fully connected which has the number of different patterns
    # as the input size and then the hidden layer and then
    # than another hidden layer
    # the output size must be the number of different classes and
    # then we apply the softmax and get probabilities for each classes
    # create three linear layers
    # by saying self l1 equals nn.linear which gets the input size and
    # then the second layer has the hidden size as input
    # the hidden size also is out put and the third layer has the hidden size as
    # input and the number of classes as output
    # input size and nomber of classes must be fixed
    # hidden size can be changed if needed
    # create an activation function
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()

    # import the  forward() or implement the forward pass so it gets self and x
    # set out to self.li and it gets x first
    # apply out activation function

    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        out = self.relu(out)
        out = self.l3(out)
        # no activation and no softmax at the end because we apply cross-entropy laws
        # this applies it for us
        # now we just return output of model
        return out