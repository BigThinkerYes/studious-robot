import numpy as np
import random
import json

# create a pytorch dataset
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from nltk_utils import bag_of_words, tokenize, stem
# import model
from model import NeuralNet

with open('intents.json', 'r') as f:
    intents = json.load(f)
    # step 1 print to view intents.json file in terminal
    # print(intents)

# step 2 comment out the print above and type the following
all_words = []
tags = []
xy = []
# loop through each sentence in our intents patterns
for intent in intents['intents']:
    # grab the tag name from the intents.json file and loop through the patterns array
    tag = intent['tag']
    # add to tag list
    tags.append(tag)

    # steps to take bellow
    # tokenize the pattern and put it in the all_words array
    # use extend to append array
    # next append with corresponding tag
    for pattern in intent['patterns']:
        # tokenize each word in the sentence
        w = tokenize(pattern)
        # add to our words list
        all_words.extend(w)
        # add to xy pair
        xy.append((w, tag))

# stem and lower each word
# define things to ignore below
ignore_words = ['?', '.', '!']
# test if ignore_words work with print below
# print(all_words)

#apply stemming change all words to lowercase
all_words = [stem(w) for w in all_words if w not in ignore_words]
# remove duplicates and sort
# print(all_words)
# get all unique words and convert to set to remove duplicate elements
# the sorted function will return a list again
# do this for all_words and tags
all_words = sorted(set(all_words))
tags = sorted(set(tags))
# test to see if we get all the tags -- used to test the size
# print(tags)

print(len(xy), "patterns")
print(len(tags), "tags:", tags)
print(len(all_words), "unique stemmed words:", all_words)

# create training data
# create a bag_of_words with a list
# x_train is a empty list and y_train is a list of tags
# y_train will be associated with a number for each tag
X_train = []
y_train = []

# loop over the xy array to get a bag_of_words
# using the tokenized values and append to x_train
for (pattern_sentence, tag) in xy:
    # X: bag of words for each pattern_sentence
    bag = bag_of_words(pattern_sentence, all_words)
    X_train.append(bag)
    # y: PyTorch CrossEntropyLoss needs only class labels, not one-hot
    # implement function for y_train
    # give each label a number as an identifier
    label = tags.index(tag)
    # CrosssEntropyLoss
    y_train.append(label)

# convert to numpy array for training data
X_train = np.array(X_train)
y_train = np.array(y_train)

# Hyper-parameters
num_epochs = 1000
batch_size = 8
learning_rate = 0.001
# input size is the length of bag_of_words as all_words
input_size = len(X_train[0])
# define model
hidden_size = 8
output_size = len(tags)
print(input_size, output_size)

# create a pytorch dataset
# create class and call the chat data set
# the class must inherit data set
# implement the init function
# store number of samples equal the length of x_train
# store data with self.x_data = x_train
# store data self.y_data = y_train


class ChatDataset(Dataset):

    def __init__(self):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = y_train

    # support indexing such that dataset[i] can be used to get i-th sample
    # implement get item function
    # access data set with an index
    # dataset[idx]
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    # we can call len(dataset) to return the size
    # define a length method
    # return number of samples
    def __len__(self):
        return self.n_samples

# test to see if we get the correct output size
# print(input_size, len(all_words))
# print(output_size, tags)

# reference ChatDataset class above


dataset = ChatDataset()
train_loader = DataLoader(dataset=dataset,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=0)

# check if gpu is available if not use cpu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# create model for model.py and push model to device
model = NeuralNet(input_size, hidden_size, output_size).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model -- loop
for epoch in range(num_epochs):
    for (words, labels) in train_loader:
        words = words.to(device)
        labels = labels.to(dtype=torch.long).to(device)

        # Forward pass
        outputs = model(words)
        # if y would be one-hot, we must apply
        # labels = torch.max(labels, 1)[1]
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # every 100th step print as an F
    # print current epoch which is the epoch plus 1
    # printing all epochs and number of epochs and
    # set loss equal to loss.item and print (.4f)four decimal values
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')
# print loss at the end
print(f'final loss: {loss.item():.4f}')

# save the data
data = {
    "model_state": model.state_dict(),
    "input_size": input_size,
    "hidden_size": hidden_size,
    "output_size": output_size,
    "all_words": all_words,
    "tags": tags
}

# define a file name
# pth is short for pytorch
# torch.save() --- will serialize and save it to a pickled file
# add the data we want to save by using the word data which is defined above
FILE = "data.pth"
torch.save(data, FILE)

# FILE saves in the same folder
print(f'training complete. file saved to {FILE}')
