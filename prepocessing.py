from typing import Counter
import numpy as np
from keras.preprocessing import sequence
from nltk.corpus import gutenberg
from string import punctuation
import nltk
from underthesea import word_tokenize
from collections import defaultdict
import re
from underthesea.word_tokenize.regex_tokenize import tokenize

def count_comma(sent):
    return sent.count(",")

def count_word(string):
    return(len(string.split()))

def cleaning(raw_texts):
    '''
    Clean other punct, and other simple processs
    '''
    data = []
    for sent in raw_texts:
        sent = sent.replace('\n','')
        sent = sent.replace(':',',')
        sent = sent.replace('!','.')
        sent = sent.replace('?','.')
        sent = sent.replace(';',',')
        sent = sent.replace('"','')
        sent = sent.replace(')','')
        sent = sent.replace('(','')
        sent = sent.replace('“','')
        sent = sent.replace('”','')
        sent = sent.replace('-','')
        sent = sent.replace('_','')
        sent = sent.replace('+','')
        sent = sent.replace('=','')
        sent = sent.replace('[','')
        sent = sent.replace(']','')
        sent = sent.replace('{','')
        sent = sent.replace('}','')
        sent = sent.replace('*','')
        sent = sent.replace('&','')
        sent = sent.replace('^','')
        sent = sent.replace('%','')
        sent = sent.replace('$','')
        sent = sent.replace('#','')
        sent = sent.replace('@','')
        sent = sent.replace('!','')
        sent = sent.replace('`','')
        sent = sent.replace('~','')
        sent = sent.replace('/','')
        sent = sent.replace('|','')
        sent = sent.replace('…','')
        sent = sent.replace(',.','')
        if count_comma(sent) >= 2:
            if count_word(sent) >= 20 and count_word(sent) <= 50:
                sent = word_tokenize(sent)
                sent = [ele.lower() for ele in sent]
                sent = ' '.join(sent)
                sent = re.sub(r"\d+", "num", sent)  
                sent = sent+'\n'
                data.append(sent)
    return data

def cleaning_test(raw_texts):
    '''
    Clean other punct, and other simple processs
    '''
    data = []
    for sent in raw_texts:
        sent = sent.replace('\n','')
        sent = sent.replace(':',',')
        sent = sent.replace('!','.')
        sent = sent.replace('?','.')
        sent = sent.replace(';',',')
        sent = sent.replace('"','')
        sent = sent.replace(')','')
        sent = sent.replace('(','')
        sent = sent.replace('“','')
        sent = sent.replace('”','')
        sent = sent.replace('-','')
        sent = sent.replace('_','')
        sent = sent.replace('+','')
        sent = sent.replace('=','')
        sent = sent.replace('[','')
        sent = sent.replace(']','')
        sent = sent.replace('{','')
        sent = sent.replace('}','')
        sent = sent.replace('*','')
        sent = sent.replace('&','')
        sent = sent.replace('^','')
        sent = sent.replace('%','')
        sent = sent.replace('$','')
        sent = sent.replace('#','')
        sent = sent.replace('@','')
        sent = sent.replace('!','')
        sent = sent.replace('`','')
        sent = sent.replace('~','')
        sent = sent.replace('/','')
        sent = sent.replace('|','')
        sent = sent.replace('…','')
        sent = sent.replace(',.','')
       
        sent = word_tokenize(sent)
        sent = [ele.lower() for ele in sent]
        sent = ' '.join(sent)
        sent = re.sub(r"\d+", "num", sent)  
        sent = sent+'\n'
        data.append(sent)
    
    return data

def create_label(text):

    '''
    Take a string -> intext and label
    '''
    tokens = word_tokenize(text)
    words = []
    ids_punct = {',':[], '.':[]}
    i = 0
    for token in tokens:
        if token not in ids_punct.keys():
            words.append(token)
            i+=1
        else:
            ids_punct[token].append(i-1)

    label = [0]*len(words)
    for pun, ids in ids_punct.items():
        for index in ids:
            label[index] = 1 if pun == ',' else 2
    
    in_text = '<fff>'.join(words)
    return in_text, label



def create_vocab(texts, topk):
    freq = Counter()
    for text in texts:
        tokens = word_tokenize(text)
        freq.update(tokens)
    most_5k = freq.most_common(topk)
    most_5k = [ele[0] for ele in most_5k]
    most_5k = ['unk'] + most_5k

    with open("./demo_data/vocab.txt", 'w') as f:
        for ele in most_5k:
            f.write(ele)
            f.write('\n')
        

    news = []
    for text in texts:
        tokens = word_tokenize(text)
        new = []
        for token in tokens:
            if token not in most_5k:
                new.append("unk")
            else:
                new.append(token)
        if new.count("unk") > 5:
            continue
        news.append(" ".join(new))
    return news


def preprocessing_train_data(RAW_PATH = './demo_data/mid_text.txt', IN_TEXT_PATH = './demo_data/fixtext.txt', LABEL_PATH = './demo_data/fixlabel.txt'):
    # start processing
    with open(RAW_PATH, 'r') as f:
        lines = f.read().splitlines()


    lines = cleaning(lines)
    lines = create_vocab(lines, topk=5000)
    texts, labels = [], []
    for text in lines:
        in_text, label = create_label(text)
        texts.append(in_text)
        labels.append(label)


    with open(IN_TEXT_PATH, 'w') as f:
        for text in texts:
            f.write(text)
            f.write('\n')

    with open(LABEL_PATH, 'w') as f:
        for label in labels :
            label = [str(ele) for ele in label]
            label = ' '.join(label)
            f.write(label)
            f.write('\n')


if __name__ == '__main__':
    preprocessing_train_data()