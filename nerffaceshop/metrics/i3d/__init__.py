import os

from .pytorch_i3d import InceptionI3d

CHECKPOINT_PATH = os.path.join(os.path.dirname(__file__), 'i3d_pretrained_400.pt')