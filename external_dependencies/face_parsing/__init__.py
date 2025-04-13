import os
import torch
from .model import BiSeNet
from torchvision import transforms

class FaceParsing(object):
    def __init__(self, device: torch.device):
        super().__init__()
        self.face2seg_ = BiSeNet(n_classes=19).to(device).eval()
        self.face2seg_.load_state_dict(torch.load(os.path.join(os.path.dirname(__file__), "../data/79999_iter.pth"), map_location=device))
        self.face2seg = lambda x: self.face2seg_(transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))(x))[0]
    def mask(self, img: torch.Tensor, seg: torch.Tensor, mask_idx: int=0):
        mask = seg == mask_idx
        return ((img/2+.5)*(~mask)*2-1)
    def __call__(self, img: torch.Tensor, argmax: bool=False) -> torch.Tensor:
        probs = self.face2seg(img)
        if argmax:
            probs = torch.argmax(probs, dim=1, keepdim=True)
        return probs