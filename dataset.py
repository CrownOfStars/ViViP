from sklearn import model_selection
import torch.nn.functional as F
import os
import os.path as osp
from decord import VideoReader,cpu,bridge
from pytorchvideo import transforms as vtransforms
import torchvision
import numpy as np
from torch.utils.data import Dataset,DataLoader,RandomSampler
import torch
import pandas as pd

bridge.set_bridge('torch')

def listpath(dirname,with_name=False):
    if with_name:
        return [(file,os.path.join(dirname,file)) for file in os.listdir(dirname)]
    else:
        return [os.path.join(dirname,file) for file in os.listdir(dirname)]


def make_dataset(dataset):
    video_infos,labels = dataset    
    return [(info,label) for info,label in zip(video_infos,labels)]

def RandomNoise(img):
    mask = torch.randn(img.shape[:2]).ge(1).unsqueeze(2).repeat((1,1,3))
    return img.masked_fill(mask,0)

class VideoDataSet(Dataset):
    def __init__(self,clips,max_interval=4,num_frames=8,transform = None):
        super(VideoDataSet, self).__init__()
        self.clips = clips
        self._max_interval = max_interval
        self._num_frames = num_frames
        self.transform = transform
        assert transform

    def __getitem__(self, index):
        path,category =self.clips[index]
        vr =  VideoReader(path,ctx=cpu(0),num_threads=2)
        if len(vr) < self._num_frames:
            x = torch.from_numpy(np.array([i for i in range(len(vr))]+[np.random.randint(0,len(vr)) for _ in range(self._num_frames-len(vr))]))
            x = x.sort().values
        else:
            interval = min(self._max_interval,len(vr)//self._num_frames)
            indices = np.linspace(0,interval*(self._num_frames-1),self._num_frames).astype(np.int16)
            x = indices + np.random.randint(0,1+len(vr)-interval*self._num_frames)
        imgs = self.transform(vr.get_batch(x).permute(3,0,1,2))
        return imgs,category

    def __len__(self):
        return len(self.clips)
    
class AddNoise(object):
    # 噪声
    def __init__(self, rate=0):
        self.rate = rate
 
    def __call__(self, img):
        mask = torch.randn(img.shape[1:]).ge(0).unsqueeze(0).repeat((3,1,1,1))
        return img.masked_fill(mask,0)

class MyTransform(object):
    def __init__(self):
        min_scale,max_scale = 256,320
        crop_size = 224
        self.worker = torchvision.transforms.Compose([
            vtransforms.ConvertUint8ToFloat(),
            vtransforms.RandomShortSideScale(min_scale,max_scale),
            vtransforms.RandomResizedCrop(crop_size,crop_size,[1.0,1.0],[1.0,1.0]),
            vtransforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
        ])
    def __call__(self,images):
        return self.worker(images)

    
    
def get_dataloader(arg):

    path = arg.dataset_path
    if arg.dataset == "UCF-101" or arg.dataset == "HMDB-51":
        
        category_list = os.listdir(path)
        train_path2videos = []
        train_labels = []
        valid_path2videos = []
        valid_labels = []

        for category,category_path in listpath(path,True):
            _video_infos = [video_path for video_path in listpath(category_path)]
            _labels = F.one_hot((torch.zeros(len(listpath(category_path)))+category_list.index(category)).long(),arg.num_classes).float()
            X_train, X_valid, Y_train, Y_valid = model_selection.train_test_split(_video_infos, _labels, test_size = 0.2)
            train_path2videos+=X_train
            valid_path2videos+=X_valid
            train_labels += Y_train
            valid_labels += Y_valid
        trainset = make_dataset([train_path2videos,train_labels])
        validset = make_dataset([valid_path2videos,valid_labels])

    else:
        raise NotImplementedError("no such dataset")

    train_dataset = VideoDataSet(clips=trainset,num_frames=arg.num_frames,transform = MyTransform())
    valid_dataset = VideoDataSet(clips=validset,num_frames=arg.num_frames,transform = MyTransform())

    train_loader = DataLoader(train_dataset,batch_size=arg.batch_size,num_workers=arg.num_workers,sampler=RandomSampler(train_dataset,replacement=True))
    valid_loader = DataLoader(valid_dataset,batch_size=arg.batch_size,num_workers=arg.num_workers,shuffle=True)

    return train_loader,valid_loader

if __name__ == "__main__":
    import argparse
    from tqdm import tqdm

    parser = argparse.ArgumentParser()
    
    parser.add_argument('--model', default='ViViP', type=str)
    parser.add_argument('--pool_model',default='s24',type=str)
    parser.add_argument('--num_heads',default=12,type=int)
    parser.add_argument('--depth',default=12,type=int)
    parser.add_argument('--dropout',default=0.5,type=float)


    parser.add_argument('--dataset', default='UCF-101', type=str)
    parser.add_argument('--dataset_path',default='/root/autodl-tmp/UCF-101',type=str)
    parser.add_argument('--num_classes',default=101,type=int)
    parser.add_argument('--num_frames',default=16,type=int)
    parser.add_argument('--num_workers',default=12,type=int)
    parser.add_argument('--batch_size',default=12, type= int)

    parser.add_argument('--optim',default='MileStones SGD',type=str)
    parser.add_argument('--start_lr',default=1e-3,type=float)
    parser.add_argument('--weight_decay',default=5e-4,type=float)
    parser.add_argument('--momentum',default=0.9,type=float)
    parser.add_argument('--gamma',default=0.2,type=float)
    parser.add_argument('--max_epoch',default=80,type=int)
    parser.add_argument('--milestones',default="[15,25,35,45,55,65,75]",type=str)

    
    parser.add_argument('--output_dir',default='./output',type=str)
    args = parser.parse_args()


    train_dl,val_dl = get_dataloader(args)
    for img,label in tqdm(train_dl):

        print(img.shape,label.shape)