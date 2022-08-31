from omegaconf import DictConfig, OmegaConf
from hydra import compose
import hydra
import urllib
from PIL import Image
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

@hydra.main(version_base=None, config_path="conf", config_name="config")
def execute(cfg: DictConfig) -> None:
    model = cfg.model
    image = cfg.image
    import timm
    model = timm.create_model(model, pretrained=True)
    model.eval()
    import json
    print((infer( model , image)))

def infer(model, image):
    config = resolve_data_config({}, model=model)
    transform = create_transform(**config)
    parts = image.split("/")
    fname = parts[-1]
    url, filename = (image, fname)
    urllib.request.urlretrieve(url, filename)
    img = Image.open(filename).convert('RGB')
    tensor = transform(img).unsqueeze(0) # transform and add batch dimension
    import torch
    with torch.no_grad():
        out = model(tensor)
    probabilities = torch.nn.functional.softmax(out[0], dim=0)
    # Get imagenet class mappings
    url, filename = ("https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt", "imagenet_classes.txt")
    urllib.request.urlretrieve(url, filename) 
    with open("imagenet_classes.txt", "r") as f:
        categories = [s.strip() for s in f.readlines()]
    # Print top categories per image
    top1_prob, top1_catid = torch.topk(probabilities, 1)
    response = ""
    import json
    for i in range(top1_prob.size(0)):
        ##output = {}
        response = f'"predicted" :  "{categories[top1_catid[i]]}", "confidence" : {top1_prob[i].item()}'
        ##response = output
    return "{"+response    +"}"
if __name__ == "__main__":
    execute()
