'''
This is a modified and adapted version of the code from
https://christianjmills.com/posts/pytorch-train-image-classifier-timm-hf-tutorial/
'''
import streamlit as st

import multiprocessing
import math
import json
from pathlib import Path
import datetime
from typing import Dict, Tuple
import argparse


import timm
from timm.data import ImageDataset as IDataset

from datasets import load_dataset

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.amp import autocast
from torch.cuda.amp import GradScaler
from torcheval.metrics import BinaryAccuracy, MulticlassAccuracy, BinaryNormalizedEntropy
from torchvision import transforms
import torchvision.transforms.functional as TF
import torchvision.transforms as T
from torch import Tensor



from tqdm.auto import tqdm


#parser = argparse.ArgumentParser(description="Train image classifier")
#parser.add_argument("--epochs", type=int, required=True)


#args = parser.parse_args()




class ImageDataset(Dataset):
    """A PyTorch Dataset class to be used in a DataLoader to create batches.
    
    Attributes:
        dataset: A list of dictionaries containing 'label' and 'image' keys.
        classes: A list of class names.
        tfms: A torchvision.transforms.Compose object combining all the desired transformations.
    """
    def __init__(self, dataset, classes, tfms):
        self.dataset = dataset
        self.classes = classes
        self.tfms = tfms
        
    def __len__(self):
        """Returns the total number of samples in this dataset."""
        return len(self.dataset)

    def __getitem__(self, idx):
        """Fetches a sample from the dataset at the given index.
        
        Args:
            idx: The index to fetch the sample from.
            
        Returns:
            A tuple of the transformed image and its corresponding label index.
        """
        sample = self.dataset[idx]
        image = sample['image'].convert("RGB")
        #label = sample['label']
        return self.tfms(image)#, label
class ResizePad(nn.Module):
    def __init__(self, max_sz=256, padding_mode='edge'):
        """
        A PyTorch module that resizes an image tensor and adds padding to make it a square tensor.

        Args:
        max_sz (int, optional): The size of the square tensor.
        padding_mode (str, optional): The padding mode used when adding padding to the tensor.
        """
        super().__init__()
        self.max_sz = max_sz
        self.padding_mode = padding_mode
        
    def forward(self, x):
        w, h = TF.get_image_size(x)
        
        size = int(min(w, h) / (max(w, h) / self.max_sz))
        x = TF.resize(x, size=size, antialias=True)
        
        w, h = TF.get_image_size(x)
        offset = (self.max_sz - min(w, h)) // 2
        padding = [0, offset] if h < w else [offset, 0]
        x = TF.pad(x, padding=padding, padding_mode=self.padding_mode)
        x = TF.resize(x, size=[self.max_sz] * 2, antialias=True)
        
        return x
class CustomTrivialAugmentWide(transforms.TrivialAugmentWide):
    def _augmentation_space(self, num_bins: int) -> Dict[str, Tuple[Tensor, bool]]:
        
        # Define custom augmentation space
        custom_augmentation_space = {
            "Identity": (torch.tensor(0.0), False),
            
            # Distort the image along the x or y axis, respectively.
            "ShearX": (torch.linspace(0.0, 0.25, num_bins), True),
            "ShearY": (torch.linspace(0.0, 0.25, num_bins), True),

            "TranslateX": (torch.linspace(0.0, 32.0, num_bins), True),
            "TranslateY": (torch.linspace(0.0, 32.0, num_bins), True),

            "Rotate": (torch.linspace(0.0, 45.0, num_bins), True),

            "Brightness": (torch.linspace(0.0, 0.75, num_bins), True),
            "Color": (torch.linspace(0.0, 0.99, num_bins), True),
            "Contrast": (torch.linspace(0.0, 0.99, num_bins), True),
            "Sharpness": (torch.linspace(0.0, 0.99, num_bins), True),

            # Reduce the number of bits used to express the color in each channel of the image.
            "Posterize": (8 - (torch.arange(num_bins) / ((num_bins - 1) / 6)).round().int(), False),

            # Invert all pixel values above a threshold.
            "Solarize": (torch.linspace(255.0, 0.0, num_bins), False),

            "AutoContrast": (torch.tensor(0.0), False),

            "Equalize": (torch.tensor(0.0), False),
        }
        
        return custom_augmentation_space




def run_epoch(model, dataloader, optimizer, metric, lr_scheduler, device, scaler, is_training):
    model.train() if is_training else model.eval()
    
    metric.reset()
    epoch_loss = 0
    progress_bar = tqdm(total=len(dataloader), desc="Train" if is_training else "Eval")

    for batch_id, sample in enumerate(dataloader):

        progressbar.progress(round(batch_id/len(dataloader)*100))
        
        inputs = sample
        inputs = inputs.to(device)
        targets = torch.tensor([[0] for _ in inputs])
        targets = targets.to(device)
        
        with torch.set_grad_enabled(is_training):
            with autocast(device):
                outputs = model(inputs)
                loss = torch.nn.functional.cross_entropy(outputs, targets.float())
        metric.update(outputs.detach().cpu(), targets.reshape(-1).detach().cpu())
        
        if is_training:
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            
            optimizer.zero_grad()
            lr_scheduler.step()
        
        loss_item = loss.item()
        epoch_loss += loss_item

        progress_bar.set_postfix(accuracy=metric.compute().item(), 
                                 loss=loss_item, 
                                 avg_loss=epoch_loss/(batch_id+1), 
                                 lr=lr_scheduler.get_last_lr()[0] if is_training else "")
        progress_bar.update()
        
        if math.isnan(loss_item) or math.isinf(loss_item):
            break
        
    progress_bar.close()
    return epoch_loss / (batch_id + 1)


def train_loop(model, train_dataloader, valid_dataloader, optimizer, metric, lr_scheduler, device, epochs, use_amp, checkpoint_path):
    scaler = GradScaler() if use_amp else None
    best_loss = float('inf')

    # Iterate over each epoch
    for epoch in tqdm(range(epochs), desc="Epochs"):
        train_loss = run_epoch(model, train_dataloader, optimizer, metric, lr_scheduler, device, scaler, is_training=True)
        
        with torch.no_grad():
            valid_loss = run_epoch(model, valid_dataloader, None, metric, None, device, scaler, is_training=False)
        
        if valid_loss < best_loss:
            best_loss = valid_loss
            metric_value = metric.compute().item()
            torch.save(model.state_dict(), checkpoint_path)
            
            training_metadata = {
                'epoch': epoch,
                'train_loss': train_loss,
                'valid_loss': valid_loss, 
                'metric_value': metric_value,
                'learning_rate': lr_scheduler.get_last_lr()[0],
                'model_architecture': model.name
            }
            
            with open(Path(checkpoint_path.parent/'training_metadata.json'), 'w') as f:
                json.dump(training_metadata, f)

        if any(math.isnan(loss) or math.isinf(loss) for loss in [train_loss, valid_loss]):
            print(f"Loss is NaN or infinite at epoch {epoch}. Stopping training.")
            break

    if use_amp:
        torch.cuda.empty_cache()

def model_train(epochs):
    train_label = st.text("Training...")
    global progressbar
    progressbar = st.progress(0)

    source_dir = str(Path(__file__).resolve().parent.parent)
    
    device = "cpu"
    dtype = torch.float32

    mean, std = ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    norm_stats = (mean, std)

    trivial_aug = CustomTrivialAugmentWide()

    train_sz = (288,288)
    resize_pad = ResizePad(max_sz=max(train_sz))

    train_tfms = transforms.Compose([
        ResizePad(max_sz=max(train_sz)),
        trivial_aug,
        transforms.ToTensor(),
        transforms.Normalize(*norm_stats),
    ])
    val_tfms = transforms.Compose([
        ResizePad(max_sz=max(train_sz)),
        transforms.ToTensor(),
        transforms.Normalize(*norm_stats),
    ])

    dataset_classes = IDataset(f"{source_dir}/images")
    class_names = list(dataset_classes.reader.class_to_idx.keys())
    #print(class_names)


    num_workers = multiprocessing.cpu_count()

    dataset = load_dataset("imagefolder", data_dir=f"{source_dir}/images/")
    train_split = dataset["train"]
    val_split = dataset["validation"]
    dataset_train = ImageDataset(dataset=train_split, classes=class_names, tfms=train_tfms)
    dataset_val = ImageDataset(dataset=val_split, classes=class_names, tfms=val_tfms)


    model_name = "efficientnet_b0.ra_in1k"

    model = timm.create_model(model_name, pretrained=True, num_classes=1)
    model = model.to(device=device, dtype=dtype)
    model.device = device
    model.name = model_name
    model.labels = class_names
    #model.load_state_dict(torch.load(source_dir+"/"+mod, map_location=torch.device("cpu")))
    
    #num_in_features = model.classifier.in_features
    #model.classifier = nn.Sigmoid()
    #print(model.classifier)
    bs = 64

    data_loader_params = {
        'batch_size': bs,
        'num_workers': num_workers,
        'persistent_workers': True,
        'pin_memory': False,
        'pin_memory_device': device,
    }

    train_dataloader = DataLoader(dataset_train, **data_loader_params, shuffle=True)
    valid_dataloader = DataLoader(dataset_val, **data_loader_params)
    project_dir = f"{source_dir}/PetClassifier"
    
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    checkpoint_dir = Path(f"{project_dir}/{timestamp}")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir/f"{model.name}.pth"
    #print(checkpoint_path)


    lr = 1e-3

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, eps=1e-5)
    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, 
                                                    max_lr=lr, 
                                                    total_steps=epochs*len(train_dataloader))
    metric = MulticlassAccuracy()


    use_amp = torch.cuda.is_available()
    #print(use_amp)
    train_loop(model, train_dataloader, valid_dataloader, optimizer, metric, lr_scheduler, device, epochs, use_amp, checkpoint_path)
    progressbar.empty()
    train_label.empty()


if __name__ == '__main__':
    model_train(5)    