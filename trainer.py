import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
from datasets import dataset_batch_iter

nll_loss = nn.NLLLoss()


class Trainer():
    def __init__(self, model, optimizer, device, save_model_path, log_path):
        self.device = device
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.num_categories = self.model.output_size
        self.save_model_path = save_model_path
        self.log_path = log_path
        #self.writer = SummaryWriter(log_dir=name_log_dir)

    def train(self, train_dataset, test_dataset, batch_size, start_epoch, end_epoch):
        self.model.train()
        # training
        for epoch in range(start_epoch, end_epoch+1):

            hidden = self.model.init_hidden(batch_size)
            hidden = [
                Variable(hidden[0].data, requires_grad=True).to(self.device),
                Variable(hidden[1].data, requires_grad=True).to(self.device)
            ]
            train_loss = 0.
            num_batch = 0
            for batch, data in enumerate(dataset_batch_iter(train_dataset, batch_size)):
                input_tensor = torch.Tensor(data['data']).type(
                    torch.LongTensor).to(self.device)
                target_tensor = torch.Tensor(data['label']).type(
                    torch.LongTensor).to(self.device)

                output, hidden = self.model(input_tensor, hidden)

                hidden = [
                    Variable(hidden[0].data, requires_grad=True).to(
                        self.device),
                    Variable(hidden[1].data, requires_grad=True).to(
                        self.device)
                ]
                #hidden = Variable(hidden.data, requires_grad=True).to(self.device)

                loss = nll_loss(output.view(-1, self.num_categories),
                                target_tensor.view(-1))

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()
                num_batch = batch+1

            if epoch % 1 == 0:
                # shuffle evaluate ??
                test_score = self.cal_score(test_dataset)

                train_score = self.cal_score(train_dataset)

                # self.writer.add_scalars('Loss',{"train_loss": train_loss, "test_loss": test_loss}, epoch)
                # self.writer.add_scalars("Accuracy", {"train _acc": train_accuracy, "test_acc":test_accuracy}, epoch)

                with open(self.log_path+'/train_score.txt', 'a') as f:
                    f.write(str(train_score))
                    f.write('\n')
                with open(self.log_path+'/test_score.txt', 'a') as f:
                    f.write(str(test_score))
                    f.write('\n')

                print(
                    f"\nEpoch {epoch} -- train loss: {train_score[0]} -- test loss: {test_score[0]} -- train acc: {train_score[1]} -- test acc: {test_score[1]}\n")

                # saving the last
                self.saving(train_dataset, epoch)

            else:
                train_loss /= num_batch
                print(f"\nEpoch {epoch} -- train loss: {train_loss}\n")

    def saving(self, train_dataset, epoch):
        filename = self.save_model_path
        id2word = train_dataset.id2word
        word2id = train_dataset.word2id
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "id2word": id2word,
            "word2id": word2id
        }, filename)
        print(f'\nSaving model successfully at epoch {epoch}\n')

    def evaluate(self, valid_dataset):
        batch_size = 64
        self.model.eval()
        test_loss = 0.
        hidden = self.model.init_hidden(batch_size)
        hidden = [
                    Variable(hidden[0].data, requires_grad=True).to(self.device),
                    Variable(hidden[1].data, requires_grad=True).to(self.device)
            ]
        correct = 0

        for batch, data in enumerate(dataset_batch_iter(valid_dataset, batch_size)):

            input_tensor = torch.Tensor(data['data']).type(
                torch.LongTensor).to(self.device)
            target_tensor = torch.Tensor(data['label']).type(
                torch.LongTensor).to(self.device)

            output, hidden = self.model(input_tensor, hidden)
            hidden = [
                    Variable(hidden[0].data, requires_grad=True).to(self.device),
                    Variable(hidden[1].data, requires_grad=True).to(self.device)
            ]
            #hidden = Variable(hidden.data, requires_grad=True).to(self.device)
            prediction = output.argmax(dim=-1)

            loss = nll_loss(output.view(-1, self.num_categories),
                            target_tensor.view(-1))

            test_loss += loss.item()

            correct += torch.sum(prediction == target_tensor).item()

        length = valid_dataset.length

        #print("batch=", batch)
        accuracy = correct/(batch_size*length*(batch+1))
        test_loss = test_loss/(length*(batch+1))

        return test_loss, accuracy

    def load(self, epoch):
        try:
            filename = self.save_model_path

            checkpoint = torch.load(filename)

            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            print(f'\n Loading model successfully at epoch {epoch}\n')
        except:
            print(f'\n Loading model fail at epoch {epoch}\n')

    def cal_score(self, valid_dataset):
        self.model.eval()
        test_loss = 0.
        batch_size = 32
        hidden = self.model.init_hidden(batch_size)

        hidden = [
            Variable(hidden[0].data, requires_grad=True).to(self.device),
            Variable(hidden[1].data, requires_grad=True).to(self.device)
        ]

        cnf_matrix = np.zeros((4, 4))
        b = 0
        for batch, data in enumerate(dataset_batch_iter(valid_dataset, batch_size)):
            b += 1

            input_tensor = torch.tensor(data['data']).type(
                torch.LongTensor).to(self.device)
            target_tensor = torch.tensor(data['label']).type(
                torch.LongTensor).to(self.device)

            output, hidden = self.model(input_tensor, hidden)

            hidden = [
                    Variable(hidden[0].data, requires_grad=True).to(self.device),
                    Variable(hidden[1].data, requires_grad=True).to(self.device)
            ]

            prediction = output.argmax(dim=-1)

            loss = nll_loss(output.view(-1, self.num_categories),
                            target_tensor.view(-1))

            test_loss += loss.item()

            for t, p in zip(target_tensor.view(-1), prediction.view(-1)):
                cnf_matrix[t.cpu().long(), p.cpu().long()] += 1

        test_loss /= b
        accuracy = np.diagonal(cnf_matrix).sum()/cnf_matrix.sum()

        precision_1 = cnf_matrix[1][1]/cnf_matrix[:, 1].sum()
        recall_1 = cnf_matrix[1][1]/cnf_matrix[1, :].sum()
        precision_2 = cnf_matrix[2][2]/cnf_matrix[:, 2].sum()
        recall_2 = cnf_matrix[2][2]/cnf_matrix[2, :].sum()

        return test_loss, accuracy, precision_1, recall_1, precision_2, recall_2
