
import torch
from torch import nn
import toolkit.slowfastnet as slowfastnet
from toolkit.misc import *

class SlowFast(nn.Module):
    def __init__(self,num_classes=101):
        super().__init__()
        self.net = slowfastnet.resnet50(class_num=num_classes)

    def forward(self,x):
        return self.net(x)

if __name__ == "__main__":

    model = SlowFast().cuda()
    
    loss_fn = nn.CrossEntropyLoss().cuda()

    imgs = torch.randn([8,3,16,224,224]).cuda()

    labels = torch.randn([8,101]).ge(0).float().cuda()

    preds = model(imgs)

    loss = loss_fn(preds,labels)

    print(loss.item())

    print(get_parameter_num(model))
    print(get_macs(model,imgs))