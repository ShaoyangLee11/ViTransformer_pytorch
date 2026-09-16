import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim

from dataset import get_ds
from model import ViTransformer,build_ViTransformer
from train import get_config

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

def validation_for_vit(num_examples=2):

    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    config=get_config('config.yaml')

    _,val_dataloader=get_ds(config)

    model=build_ViTransformer(config['img_size'],config['patch_size'],config['in_chans'],config['d_model'],config['classification'],config['N'],config['head'])
    weight_filename=latest_weights_file_path(config)
    state=torch.load(weight_filename)
    model.load_state_dict(state['model_state_dict'])

    model=model.to(device)
    
    with torch.no_grad():
        count=0
        for batch,labels in val_dataloader:
            count+=1

            batch=batch.to(device)
            labels=labels.to(device)
            
            model_output=model(batch)
            expect_num=labels.item()
            _,pred_num=torch.max(model_output,dim=-1)

            print(f'TARGET TEXT:{expect_num}')
            print(f'PREDICTED TEXT:{pred_num}')
            print('-------------------------')

            if count==num_examples:
                break

if __name__=='__main__':

    validation_for_vit(100)