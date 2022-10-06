# chat.py can be run with saved data file(data.pth)
import random
import json
import torch
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

# check if gpu or cpu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# open files
with open('intents.json', 'r') as json_data:
    intents = json.load(json_data)
# open saved file
FILE = "data.pth"
data = torch.load(FILE)

# information to create our model
input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data["all_words"]
tags = data["tags"]
model_sate = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
# load state of creation to know learned parameters
model.load_state_dict(model_sate)
# set evaluation mode
model.eval()
# implement chat
# give bot a name
# studious-robot
bot_name = "studious-robot"


def get_response(msg):
    # tokenize sentence
    sentence = tokenize(msg)
    # all words from saved file
    X = bag_of_words(sentence, all_words)
    # reshape
    X = X.reshape(1, X.shape[0])
    # bag_of_words returns a numpy array
    # convert to torch tensile
    X = torch.from_numpy(X).to(device)

    # use output
    # get prediction
    output = model(X)
    _, predicted = torch.max(output, dim=1)

    # get the tag with the class label predicted
    tag = tags[predicted.item()]

    # get probability
    # check if tag matches
    # and get random choice
    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                return random.choice(intent['responses'])

    return "I do not understand..."


if __name__ == "__main__":
    print("What's on your mind? (type 'quit' to exit)")
    while True:
        # sentence = "do you use credit cards?"
        sentence = input("You: ")
        if sentence == "quit":
            break

        resp = get_response(sentence)
        print(resp)