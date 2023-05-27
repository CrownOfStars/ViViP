import torch
import pickle
import matplotlib.pyplot as plt
import os
import os.path as osp

import torch
from datetime import datetime
import logging
import os
import os.path as osp

def get_datetime():
    cur_time = datetime.fromtimestamp(datetime.now().timestamp())
    str_date_time = cur_time.strftime("%Y-%m-%d,%H-%M-%S")
    return str_date_time

def init_logger(output_dir):
    
    formatter = logging.Formatter("%(asctime)s %(levelname)s: %(message)s",datefmt='%Y-%m-%d %X')

    logger = logging.getLogger('mylog')
    logger.setLevel(logging.DEBUG)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    osp.exists(output_dir),"{} not exist".format(output_dir)
    log_dir = osp.join(output_dir,get_datetime())
    os.mkdir(log_dir)
    file_handler=logging.FileHandler(filename=osp.join(log_dir,"log.txt"),mode='w',encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger,log_dir

class Meter:
    def __init__(self,dataloader,logger,log_dir,mode="train"):
        self.epoch_acc = 0
        self.epoch_loss = 0

        self.best_epoch_acc = 0.0

        self.logger = logger
        self.log_dir = log_dir
        self.model_path = osp.join(self.log_dir,'best_model.ckpt')
        self.mode = mode

        self.acc_list = []
        self.loss_list = []

        self.cur_batch = 0
        self.cur_epoch = 0
        
        self.loader_length = len(dataloader)
        self.dataset_length = len(dataloader.dataset)

    def update_batch(self,loss,preds,labels):
        
        temp_acc = torch.sum(preds.argmax(dim=1)==labels.argmax(dim=1))
        self.logger.debug("{} Epoch:{}\tBatch:{}\tAcc:{:.2f}%,\tLoss:{:.4f}".format(self.mode,self.cur_epoch,self.cur_batch,temp_acc*100/len(preds),loss))
        self.epoch_acc += temp_acc
        self.epoch_loss += loss
        self.cur_batch += 1
        
    def update_epoch(self,state_dict):
        
        epoch_acc = self.epoch_acc*100/self.dataset_length
        epoch_loss = self.epoch_loss/self.loader_length

        self.logger.debug('{}\tepoch{}_acc:{:.2f}%'.format(self.mode,self.cur_epoch,epoch_acc))
        self.logger.debug('{}\tepoch{}_loss{}'.format(self.mode,self.cur_epoch,epoch_loss))
        self.acc_list.append(epoch_acc)
        self.loss_list.append(epoch_loss)
        self.cur_epoch += 1
        self.cur_batch = 0
        self.epoch_acc = 0
        self.epoch_loss = 0.0
        
        if state_dict and epoch_acc >= self.best_epoch_acc:
                torch.save(open(osp.join(self.log_dir,'best_model.ckpt'),'wb'),state_dict)
                self.best_epoch_acc = self.epoch_acc    

    def save_last_result(self):

        plt.plot(self.acc_list)
        plt.savefig(osp.join(self.log_dir,f'{self.mode}_acc.png'))
        plt.clf()
        plt.plot(self.loss_list)
        plt.savefig(osp.join(self.log_dir,f'{self.mode}_loss.png'))
        plt.clf()
        
        pickle.dump(self.loss_list,open(osp.join(self.log_dir,f'{self.mode}_loss.pkl'),'wb'))
        pickle.dump(self.acc_list,open(osp.join(self.log_dir,f'{self.mode}_acc.pkl'),'wb'))

if __name__ == "__main__":
    pass