from pathlib import Path

import pandas as pd

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
    
    test_img = inp.convert("RGB")
    
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
    })
    return pred_data
    

def inference(cp, img, label):
    
    source_dir = str(Path(__file__).resolve().parent.parent)
    cp_path = source_dir+"/models/"+cp

    global device
    device = 'cpu'
    dtype = torch.float32
    
    global class_names
    global class_map
   
    class_names = ['Correct', 'Incorrect']
    class_map = {'Correct':'Correct', 'Incorrect':'Incorrect'}

    model_name = "efficientnet_b0.ra_in1k"

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

