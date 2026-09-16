import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader



def get_ds(config):

    # MNIST 传入的格式是PIL image格式,可以利用torchvision.transforms.ToTensor()转成张量
    train_dataset=torchvision.datasets.MNIST(root='./data',train=True,transform=transforms.ToTensor(),download=True)
    val_dataset=torchvision.datasets.MNIST(root='./data',train=False,transform=transforms.ToTensor(),download=True)

    train_dataloader=DataLoader(train_dataset,batch_size=config['batch_size'],shuffle=True)
    val_dataloader=DataLoader(val_dataset,batch_size=1,shuffle=False)

    return train_dataloader,val_dataloader




