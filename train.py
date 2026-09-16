import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim

from dataset import get_ds
from model import ViTransformer,build_ViTransformer


from pathlib import Path
import yaml

def latest_weights_file_path(config):
  
    model_folder = config['model_folder']
    model_basename = config['model_basename']
    model_filename = f"{model_basename}*.pt"
    weights_files = list(Path(model_folder).glob(model_filename))

    if len(weights_files) == 0:
        return None
  
    weights_files.sort(key=lambda x: int(x.stem.split('_')[-1]))
    return str(weights_files[-1])

def get_config(filename:str):

    file1=open(file=filename,mode='r')
    config_str=file1.read()
    file1.close()

    config_dict=yaml.load(config_str,Loader=yaml.FullLoader)

    return config_dict

def get_model(config):

    model=build_ViTransformer(config['img_size'],config['patch_size'],config['in_chans'],config['d_model'],config['classification'],config['N'],config['head'])

    return model

def get_weight_file(config,epoch:str):

    model_folder=config['model_folder']
    model_basename=config['model_basename']
    model_filename=f'{model_basename}{epoch}.pt'

    return str(Path('.')/model_folder/model_filename)

def train(config):

    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    Path(config['model_folder']).mkdir(parents=True,exist_ok=True)

    train_dataloader,_=get_ds(config)

    model=get_model(config)
    model=model.to(device)


    writer=SummaryWriter(config['experiment_name'])

    optimizer=optim.Adam(params=model.parameters(),lr=config['lr'])

    scheduler=optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=config['num_epochs'],eta_min=1e-6)

    initial_epoch=0
    global_step=0

    if config['preload']:
        model_filename=latest_weights_file_path(config=config)
        print(f'Preload model {model_filename}')
        state=torch.load(model_filename)
        initial_epoch=state['epoch']+1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        model.load_state_dict(state['model_state_dict'])
        global_step=state['global_step']

    loss_fn=nn.CrossEntropyLoss(label_smoothing=0.1).to(device)


    for epoch in range(initial_epoch,initial_epoch+50):

        model.train()
        batch_iterator=tqdm(iterable=train_dataloader,desc=f'Processing epoch{epoch:02d}')
        """
        train_dataloader是iterable,也就是可迭代对象
        tqdm把train_dataloader包装为迭代器iterator
        iterator存储train_dataloader中的所有数据,但每次迭代只输出一条数据(在dataloader中"一条数据"是指已经被打包成一个batch的数据,所以一次迭代就输出一个batch)

        """
        for batch,labels in batch_iterator:            # (B,C,H,W)
            model.train()

            batch=batch.to(device)
            outputs=model(batch)

            labels=labels.long().to(device)

            loss=loss_fn(outputs,labels.long())

            batch_iterator.set_postfix(loss=f'{loss.item():6.3f}')

            """
            set_postfix 作用:
            在进度条末尾追加一组键值对，实时显示指标(这里就是 loss)每一轮 batch 都会刷新。
            """

            # Log the loss

            writer.add_scalar(tag="train loss",scalar_value=loss.item(),global_step=global_step)
            writer.flush()

            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            global_step+=1

        
        scheduler.step()
        # Save the model after each epoch
        model_filename=get_weight_file(config=config,epoch=f'{epoch:02d}')
        torch.save(
            {
                'epoch':epoch,
                'model_state_dict':model.state_dict(),
                'optimizer_state_dict':optimizer.state_dict(),
                'global_step':global_step
            }
            ,model_filename
        )
if __name__=='__main__':

    config=get_config('config.yaml')
    train(config)

