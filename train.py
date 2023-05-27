from model import *
from dataset import get_dataloader
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import argparse
from meter import Meter
from mylog import init_logger

from models.SlowFast import SlowFast
from models.TSN import TSN
from models.TimeSformer import TimeSformer
from models.ViViP import ViViP
from models.ViViT import ViViT

def fprint_args(args,logger):
     for arg in vars(args):
         logger.debug("{}:{}".format(arg, getattr(args, arg)))

def train_one_epoch(train_loader, model, optimizer, meter:Meter):

    loss_fun = nn.CrossEntropyLoss()
    model.train()
    for imgs,labels in tqdm(train_loader):
        imgs = imgs.cuda(non_blocking=True)
        labels = labels.cuda()
        
        preds = model(imgs)
        loss = loss_fun(preds, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        meter.update_batch(loss.item(),preds,labels)
    meter.update_epoch(None)
        

def valid_one_epoch(valid_loader,model,meter:Meter):
    model.eval()
    loss_fun = nn.CrossEntropyLoss()
    
    with torch.no_grad():
    
        for imgs, labels in tqdm(valid_loader):

            imgs = imgs.cuda(non_blocking=True)
            labels = labels.cuda()
        
            preds = model(imgs)

            loss = loss_fun(preds, labels)
            
            meter.update_batch(loss.item(),preds,labels)
        meter.update_epoch(None)

def get_model(arg):
    if arg.model == "SlowFast":
        return SlowFast(num_classes=arg.num_classes)
    elif arg.model == "TSN":
        return TSN(num_classes=arg.num_classes)
    elif arg.model == "ViViP":
        return ViViP(num_classes=arg.num_classes,depth=arg.depth,heads=arg.heads,dropout=arg.dropout,pool_model=arg.pool_model)
    elif arg.model == "ViViT":
            return ViViT(num_classes=arg.num_classes)
    elif arg.model == "TimeSformer":
            return TimeSformer(num_classes=arg.num_classes)
    else:
        raise 
        
def train(args):

    train_dataloader,valid_dataloader = get_dataloader(args)

    model = eval(args.model)(pool_model=args.pool_model,num_classes=args.num_classes).cuda()

    optimizer = optim.SGD(model.parameters(),lr = args.start_lr,momentum=args.momentum,weight_decay=args.weight_decay)
    
    optim.lr_scheduler.MultiStepLR(optimizer,milestones=args.milestones,gamma=args.gamma,last_epoch=-1)

    logger,log_dir = init_logger(args.output_dir)
    

    train_meter = Meter(train_dataloader,logger,log_dir,"train")
    
    valid_meter = Meter(valid_dataloader,logger,log_dir,"valid")

    fprint_args(args,logger)
    
    print(model.pool_model)
    
    for _ in range(args.max_epoch):

        train_one_epoch(train_dataloader,model,optimizer,train_meter)

        valid_one_epoch(valid_dataloader,model,valid_meter)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    
    parser.add_argument('--model', default='ViViP', type=str)
    parser.add_argument('--pool_model',default='s36',type=str)
    parser.add_argument('--num_heads',default=12,type=int)
    parser.add_argument('--depth',default=12,type=int)
    parser.add_argument('--dropout',default=0.5,type=float)


    parser.add_argument('--dataset', default='UCF-101', type=str)
    parser.add_argument('--dataset_path',default='/root/autodl-tmp/UCF-101',type=str)
    parser.add_argument('--num_classes',default=101,type=int)
    parser.add_argument('--num_frames',default=16,type=int)
    parser.add_argument('--num_workers',default=12,type=int)
    parser.add_argument('--batch_size',default=10, type= int)

    parser.add_argument('--optim',default='MileStones SGD',type=str)
    parser.add_argument('--start_lr',default=1e-3,type=float)
    parser.add_argument('--weight_decay',default=5e-4,type=float)
    parser.add_argument('--momentum',default=0.9,type=float)
    parser.add_argument('--gamma',default=0.2,type=float)
    parser.add_argument('--max_epoch',default=80,type=int)
    parser.add_argument('--milestones',default="[15,25,35,45,55,65,75]",type=str)

    
    parser.add_argument('--output_dir',default='./output',type=str)
    args = parser.parse_args()

    train(args)

