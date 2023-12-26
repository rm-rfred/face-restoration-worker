import os

import torch

from .bisenet import BiSeNet
from .parsenet import ParseNet

MODELS_PATH = os.environ.get("MODELS_PATH", "/models/")


def init_parsing_model(model_name="bisenet", half=False, device="cuda"):
    if model_name == "bisenet":
        model = BiSeNet(num_class=19)
    elif model_name == "parsenet":
        model = ParseNet(in_size=512, out_size=512, parsing_ch=19)
    else:
        raise NotImplementedError(f"{model_name} is not implemented.")

    model_path = os.path.join(MODELS_PATH, "parsing_parsenet.pth")
    load_net = torch.load(model_path, map_location=lambda storage, loc: storage)
    model.load_state_dict(load_net, strict=True)
    model.eval()
    model = model.to(device)
    return model
