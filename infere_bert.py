import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import transformers
from datasets import *
from models import *
from prepocessing import *
import numpy as np
from keras.preprocessing import sequence
from transformers import AutoModel, AutoTokenizer



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Loading saved model
length = 32
embedding_size = 512
hidden_dim = 256
n_layers = 2
num_categories = 4
bert_dim = 768
MODEL_PATH = './dumps/bertPP_big.pt'


tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base", use_fast=False)
phobert = AutoModel.from_pretrained("vinai/phobert-base")
rnn = nn.RNN(bert_dim, embedding_size, n_layers, batch_first = True, dropout=0.2, bidirectional=True)
head = MLP_Head(embedding_size*2, hidden_dim, num_categories)
mymodel = My_BertBase_Model(phobert, rnn, head, device)


def load():
    try:
        checkpoint = torch.load(MODEL_PATH, map_location=device)
        mymodel.load_state_dict(checkpoint['model_state_dict'])  
        #optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print("Load successfully")
    except:
        print("Load fail")


load()


def infere(model, input_tensor):
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
    prediction = output.view(-1, 4).argmax(dim=-1)
    return prediction


def restore(words, prediction):
    if len(words) > 32:
        words = words[:32]
    convert = {0: '', 1: ',', 2: '.', 3: ''}
    seq = [ word+convert[prediction[i]] for i, word in enumerate(words)]
    seq = ' '.join(seq)
    return seq

    


def pipleline_bertbase(sentence):
    sentence = [sentence]

    cleaned = cleaning_test(sentence)[0]
    
    
    in_text, label = create_label(cleaned)
    words = in_text.split("<fff>")
    
    words = ['<s>']+words+['</s>']
    
    tokens = tokenizer.convert_tokens_to_ids(words)

    tokens = sequence.pad_sequences([tokens], maxlen=length, padding='post', value=1)[0]

    label = [int(ele) for ele in label]
    label = [0]+label+[3]
    label = sequence.pad_sequences([label], maxlen=length, padding="post", value=3)[0]


    input_tensor = torch.Tensor([tokens]).long().to(device)

    prediction = infere(mymodel, input_tensor)
    prediction = prediction.cpu().numpy()
    res = restore(words, prediction)

    return res

    

 
if __name__ == '__main__':
    sent = "Trong đó, 15 trường hợp là tiếp xúc gần với bệnh nhân đã được cách ly từ trước nên không có khả năng lây nhiễm tiếp cho cộng đồng."
    res = pipleline_bertbase(sent)
    print(res)