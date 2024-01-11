from pathlib import Path
import os

import pandas as pd

from PIL import Image

import timm


import torch
from torch import nn
from torchvision import transforms




class InferenceWrapper(nn.Module):
    def __init__(self, model, normalize_mean, normalize_std, scale_inp=False, channels_first=False):
        super().__init__()
        self.model = model
        self.register_buffer("normalize_mean", normalize_mean)
        self.register_buffer("normalize_std", normalize_std)
        self.scale_inp = scale_inp
        self.channels_first = channels_first
        self.softmax = nn.Softmax(dim=1)

    def preprocess_input(self, x):
        if self.scale_inp:
            x = x / 255.0

        if self.channels_first:
            x = x.permute(0, 3, 1, 2)

        x = (x - self.normalize_mean) / self.normalize_std
        return x

    def forward(self, x):
        x = self.preprocess_input(x)
        x = self.model(x)
        x = self.softmax(x)
        return x

def predict(target, inp, wrapped_model):
    global class_map
    global class_names
    print(inp)
    test_img = inp.convert("RGB")
    test_img.show()
    
    target_cls = target

    infer_sz = 288
    inp_img = test_img.resize((infer_sz, infer_sz))

    img_tensor = transforms.ToTensor()(inp_img)[None].to(device)

    with torch.no_grad():
        pred_scores = wrapped_model(img_tensor)

    confidence_score = pred_scores.max()

    pred_class = class_map[class_names[torch.argmax(pred_scores)]]

    pred_data = pd.Series({
        "Target": target_cls,
        "Predicted": pred_class,
        "Confidence Score": round(confidence_score.item(), 4),
        "Model": wrapped_model.model.name,
    })
    return pred_data
    

def inference(cp, img, label):
    source_dir = str(Path(__file__).resolve().parent.parent)
    cp_path = source_dir+"/"+cp

    global device
    device = 'cpu'
    dtype = torch.float32
    
    global class_names
    class_names = ['Abyssinian', 'american_bulldog', 'american_pit_bull_terrier', 'basset_hound', 'beagle', 'Bengal', 'Birman', 'Bombay', 'boxer', 'British_Shorthair', 'chihuahua', 'Egyptian_Mau', 'english_cocker_spaniel', 'english_setter', 'german_shorthaired', 'great_pyrenees', 'havanese', 'japanese_chin', 'keeshond', 'leonberger', 'Maine_Coon', 'miniature_pinscher', 'newfoundland', 'Persian', 'pomeranian', 'pug', 'Ragdoll', 'Russian_Blue', 'saint_bernard', 'samoyed', 'scottish_terrier', 'shiba_inu', 'Siamese', 'Sphynx', 'staffordshire_bull_terrier', 'wheaten_terrier', 'yorkshire_terrier']
    global class_map
    class_map = {'Abyssinian':'Abyssinian', 'english_cocker_spaniel':'american_bulldog', 'english_setter':'american_pit_bull_terrier', 'german_shorthaired':'basset_hound', 'great_pyrenees':'beagle', 'american_bulldog':'Bengal', 'american_pit_bull_terrier':'Birman', 'basset_hound':'Bombay', 'havanese':'boxer', 'beagle':'British_Shorthair', 'japanese_chin':'chihuahua', 'Bengal':'Egyptian_Mau', 'keeshond':'english_cocker_spaniel', 'leonberger':'english_setter', 'Maine_Coon':'german_shorthaired', 'miniature_pinscher':'great_pyrenees', 'newfoundland':'havanese', 'Persian':'japanese_chin', 'pomeranian':'keeshond', 'pug':'leonberger', 'Birman':'Maine_Coon', 'Ragdoll':'miniature_pinscher', 'Russian_Blue':'newfoundland', 'Bombay':'Persian', 'saint_bernard':'pomeranian', 'samoyed':'pug', 'boxer':'Ragdoll', 'British_Shorthair':'Russian_Blue', 'scottish_terrier':'saint_bernard', 'shiba_inu':'samoyed', 'Siamese':'scottish_terrier', 'Sphynx':'shiba_inu', 'chihuahua':'Siamese', 'Egyptian_Mau':'Sphynx', 'staffordshire_bull_terrier':'staffordshire_bull_terrier', 'wheaten_terrier':'wheaten_terrier', 'yorkshire_terrier':'yorkshire_terrier'}


    model_name = os.path.basename(cp).strip(".pth")

    model = timm.create_model(model_name, num_classes=len(class_names))
    model = model.to(device=device, dtype=dtype).eval()
    model.device = device
    model.name = model_name

    model.load_state_dict(torch.load(cp_path, map_location=torch.device("cpu")))

    mean, std = ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    norm_stats = (mean, std)

    normalize_mean = torch.tensor(norm_stats[0]).view(1, 3, 1, 1)
    normalize_std = torch.tensor(norm_stats[1]).view(1, 3, 1, 1)

    wrapped_model = InferenceWrapper(model, normalize_mean, normalize_std).to(device=device)
    wrapped_model.eval()

    res = []
    
    r = predict(label, img, wrapped_model)
    res.append(r)

    res = pd.DataFrame(res)
    return res

        
    
        
    

    