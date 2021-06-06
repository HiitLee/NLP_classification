
import torch
from tqdm import tqdm

class Trainer(object):
    def __init__(self, args, train_iter, test_iter, model, optimizer, criterion):
        self.train_iter = train_iter
        self.test_iter = test_iter
        self.args = args
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion

    def train_epoch(self):
        maxScore = 0
        for e in range(0, self.args.epochs):
            iter_bar = tqdm(self.train_iter, desc='Iter (loss=X.XXX)')
            loss_sum=0
            for tt, samples  in enumerate(iter_bar):
                text, text1, target, seq_lengths = samples
                text, text1, target = text.cuda(), text1.cuda(),  target.cuda()
                self.optimizer.zero_grad()
                if('HCL' not in self.args.model and 'LSTM' in self.args.model):
                    seq_lengths, perm_idx = seq_lengths[0].sort(0, descending=True)
                    text = text[perm_idx]
                    target = target[perm_idx]
                    logit = self.model(text, seq_lengths)
                else:
                    logit = self.model(text, text1)
                                                                                                                   
                loss = self.criterion(logit.cuda(), target.squeeze().cuda())
                loss.backward()
                self.optimizer.step()
                loss_sum += loss.item()
                iter_bar.set_description('Iter (loss=%5.3f)'%loss.item())
            print('Epoch %d/%d : Average Loss %5.3f'%(e+1, 100, loss_sum/(tt+1)))
            iter_bar_22 = tqdm(self.test_iter, desc='Iter (loss=X.XXX)')
            corrects, avg_loss, alls = 0, 0, 0
            for tt, samples in enumerate(iter_bar_22):
                text, text1, target, seq_lengths = samples
                text, text1, target = text.cuda(),text1.cuda(), target.cuda()
                if('HCL' not in self.args.model):
                    seq_lengths, perm_idx = seq_lengths[0].sort(0, descending=True)
                    text = text[perm_idx]
                    target = target[perm_idx]
                    logit = self.model(text, seq_lengths)
                else:
                    logit = self.model(text, text1)
                                                                                                                   
                loss = self.criterion(logit.cuda(), target.squeeze().cuda())
                _, y_pred3 = logit.max(1)
                avg_loss += loss.item()
                for i in range(0, len(y_pred3)):
                    if(y_pred3[i].item() == target[i]):
                        corrects+=1
                                                                                                                   
                iter_bar_22.set_description('Iter (accur=%5.3f)'% float(corrects/(self.args.batch*(tt+1))))
                                                                                                                   
            avg_loss = avg_loss/(tt+1)
            accuracy = 100.0 * (float(corrects/(self.args.batch*(tt+1))))
            if(maxScore < accuracy):
                maxScore = accuracy
                torch.save(self.model.state_dict(),'./save_model/model')
                                                                                                                   
            print(' Evaluation - loss: {:.6f}  acc: {:.4f}%({}/{})'.format(avg_loss, accuracy, corrects, self.args.batch*(tt+1)))

    def eval(self):
        self.model.load_state_dict(torch.load('./save_model/model'))
        iter_bar_22 = tqdm(self.test_iter, desc='Iter (loss=X.XXX)')
        corrects, avg_loss, alls = 0, 0, 0
        for tt, samples in enumerate(iter_bar_22):
            text, text1, target, seq_lengths = samples
            text, text1, target = text.cuda(),text1.cuda(), target.cuda()
            if('HCL' not in self.args.model):
                seq_lengths, perm_idx = seq_lengths[0].sort(0, descending=True)
                text = text[perm_idx]
                target = target[perm_idx]
                logit = self.model(text, seq_lengths)
            else:
                logit = self.model(text, text1)
                                                                                                               
            loss = self.criterion(logit.cuda(), target.squeeze().cuda())
            _, y_pred3 = logit.max(1)
            avg_loss += loss.item()
            for i in range(0, len(y_pred3)):
                if(y_pred3[i].item() == target[i]):
                    corrects+=1
                                                                                                               
            iter_bar_22.set_description('Iter (accur=%5.3f)'% float(corrects/(self.args.batch*(tt+1))))
                                                                                                               
        avg_loss = avg_loss/(tt+1)
        accuracy = 100.0 * (float(corrects/(self.args.batch*(tt+1))))
                                                                                                               
        print(' Evaluation - loss: {:.6f}  acc: {:.4f}%({}/{})'.format(avg_loss, accuracy, corrects, self.args.batch*(tt+1)))
