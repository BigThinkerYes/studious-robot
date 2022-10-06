import nltk
import numpy as np

# used to prefill if empty when project is started
# will automatically download files
# after downloading comment the following lines out

# nltk.download('punkt')

from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()

#util functions
#use these defined names in the train.py file


def tokenize(sentence):
    return nltk.word_tokenize(sentence)


def stem(word):
    return stemmer.stem(word.lower())


def bag_of_words(tokenized_sentence, all_words):
    # implement bag_of_words function
    """
    sentence = ["hello", "how", "are", "you"]
    words = ["hi", "hello", "I", "you", "bye", "thank", "cool"]
    bag = [ 0, 1, 0, 1, 0, 0,  0]
    """
    # create a tokenized sentence equal to our stemming function
    # IN OUR WORD w for w  in tokenized sentence to apply stemming

    tokenized_sentence = [stem(w) for w in tokenized_sentence]

    # create a array with the same size using numpy to set to 0
    # create a dataType and loop over the words

    bag = np.zeros(len(all_words), dtype=np.float32)
    # get imdex and current word
    # check if this word is in our word is in our tokenized sentence
    # with w in tokenized_sentence
    # then it will get a 1
    # set index equal to 1 as a float
    # then return the bag
    for idx, w in enumerate(all_words):
        if w in tokenized_sentence:
            bag[idx] = 1.0
    return bag

# used for testing  to make sure you get [ 0, 1, 0, 1, 0, 0,  0]
# tests line 44 through 47
# use tokenized words
# sentence = ["hello", "how", "are", "you"]
# words = ["hi", "hello", "I", "you", "bye", "thank", "cool"]
# #call the above
# bag = bag_of_words(sentence, words)
# print(bag)



#end of the defined names


# part of the second test
# test if stem works
# tests before train.py was added
# words = ["Organize", "organizes", "organizing"]
# stemmed_words = [stem(w) for w in words]
# print(stemmed_words)


# part of step one
# used to test if code works , # nltk.download('punkt') is used when no data avaliable
# a = "How long?"
# print(a)
# a = tokenize(a)
# print(a)
