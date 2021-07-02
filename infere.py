import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from datasets import *
from models import *
from prepocessing import *
import numpy as np
from keras.preprocessing import sequence



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Loading saved model
length = 32
embedding_size = 512
hidden_dim = 256
n_layers = 2
num_categories = 4
MODEL_PATH = './dumps/BiLSTM.pt'

checkpoint = torch.load(MODEL_PATH, map_location=device)
id2word = checkpoint['id2word']
word2id = checkpoint['word2id']

model = BiLSTMModel(len(id2word), embedding_size,
                 num_categories, hidden_dim, n_layers).to(device)
model.load_state_dict(checkpoint['model_state_dict'])


def infere(model, input_tensor):
    model.eval()
    batch_size = input_tensor.shape[0]
    hidden = model.init_hidden(batch_size)
    output, _ = model(input_tensor, hidden)
    prediction = output.argmax(dim=-1)
    return prediction


def restore(words, prediction):
    convert = {0: '', 1: ',', 2: '.', 3: ''}
    seq = [ word+convert[prediction[i]] for i, word in enumerate(words)]
    seq = ' '.join(seq)
    return seq

    


def pipleline(sentence):
    sentence = [sentence]

    cleaned = cleaning_test(sentence)[0]
    
    
    in_text, label = create_label(cleaned)
    words = in_text.split("<fff>")


    tokens = [word2id[word] if word in word2id else word2id['unk'] for word in words]
    tokens = sequence.pad_sequences([tokens], maxlen=length, padding='post')[0]

    label = [int(ele) for ele in label]
    label = sequence.pad_sequences([label], maxlen=length, padding="post", value=3)[0]


    input_tensor = torch.Tensor([tokens]).long().to(device)
    prediction = infere(model, input_tensor)
    prediction = prediction.cpu().view(-1).numpy()
    res = restore(words, prediction)

    return res

    

 
    