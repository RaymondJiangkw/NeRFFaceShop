import os
import sys
sys.path.append(os.path.dirname(__file__))

from typing import Union, Literal
import torch
from demo_utils import get_flame_params
from predictor import FaceMeshPredictor
from torchvision import transforms

class DAD3DHeads(object):
    def __init__(self, device):
        super().__init__()
        self.predictor = FaceMeshPredictor.dad_3dnet(device)
    def __call__(self, image: torch.Tensor, type_: Union[Literal['flame'], Literal['landmarks_2d']] = 'landmarks_2d'):
        predictions = self.predictor(image)
        if type_ == 'flame':
            return get_flame_params(predictions, image)
        elif type_ == 'landmarks_2d':
            return predictions['points'].view(-1, 68, 2) / image.shape[-1]
        else:
            raise NotImplementedError(f"`{type_}` is unable to be identified.")