import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import argparse
from datasets import MyDataset, TestDataset, dataset_batch_iter
from models import RNNModel, GRUModel, BiLSTMModel
from trainer import Trainer
from sklearn.metrics import multilabel_confusion_matrix



parser = argparse.ArgumentParser()
## config for model
parser.add_argument('--model', type=str, default='BiLSTM', help="RNN or GRU architecture")  # change it GRU or BiLSTM
parser.add_argument('--n_layer', type=int, default=2, help="Num layers of architecture")
parser.add_argument('--embedding_size', type=int, default=512, help="Embedding size of a word")  
parser.add_argument('--hidden_dim', type=int, default=256, help="hidden dim state of block RNN") 
parser.add_argument('--output_dim', type=int, default=4, help="output head dim (4 for 0, 1, 2, 3)") # do not change

## config for dataset
parser.add_argument('--train_text_path', type=str, default='./demo_data/fixtext.txt', help='train text path')
parser.add_argument('--train_label_path', type=str, default='./demo_data/fixlabel.txt', help='train label path')
parser.add_argument('--valid_text_path', type=str, default='./demo_data/testtext.txt', help='valid text path')
parser.add_argument('--valid_label_path', type=str, default='./demo_data/testlabel.txt', help='valid label path')
parser.add_argument('--length', type=int, default=32, help='max length of a sentence')       

## config for optimizer
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate ')
parser.add_argument('--betas', type=tuple, default=(0.9, 0.999), help='betas')



## config for running

parser.add_argument('--batch_size', type=int, default=32, help='Batch size training')
parser.add_argument('--saved_model_path', type=str, default='./dumps/', help='saved model path')
parser.add_argument('--logs_path', type=str, default='./logs', help='logs dir save score')

opt = parser.parse_args()
nll_loss = nn.NLLLoss()


        

if __name__ == '__main__':
    train_dataset = MyDataset(opt.train_text_path, opt.train_label_path, opt.length)
    test_dataset = TestDataset(opt.valid_text_path, opt.valid_label_path, opt.length, word2id=train_dataset.word2id, id2word=train_dataset.id2word)
    
    if opt.model == 'GRU':
        model = GRUModel(
            vocab_size=train_dataset.vocab_size,
            embedding_size=opt.embedding_size,
            output_size=opt.output_dim, 
            hidden_dim=opt.hidden_dim,
            n_layers=opt.n_layer,
        )
    elif opt.model == 'BiLSTM':
        model = BiLSTMModel(
            vocab_size=train_dataset.vocab_size,
            embedding_size=opt.embedding_size,
            output_size=opt.output_dim, 
            hidden_dim=opt.hidden_dim,
            n_layers=opt.n_layer,
            bidirectional=True
        )
    else:
        model = RNNModel(
            vocab_size=train_dataset.vocab_size,
            embedding_size=opt.embedding_size,
            output_size=opt.output_dim, 
            hidden_dim=opt.hidden_dim,
            n_layers=opt.n_layer,
        )
        
    optimizer = optim.Adam(
        model.parameters(), 
        lr=opt.lr,
        betas=opt.betas
        
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    trainer = Trainer(
        model, optimizer, device,
        save_model_path=opt.saved_model_path+opt.model+'.pt',
        log_path=opt.logs_path
    )

    vocab_size = train_dataset.vocab_size

    trainer.train(train_dataset, test_dataset, opt.batch_size, start_epoch=1, end_epoch=100)
    
    