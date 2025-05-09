import requests
from typing import Dict, Any, Tuple, Union, List

import os
dirname = os.path.dirname(__file__)
import cv2
import torch
from torch import Tensor
import numpy as np
from torchvision import transforms

import logging

from utils import load_yaml, get_relative_path
from model_training.head_mesh import HeadMesh
from model_training.model.utils import to_device, unravel_index, calculate_paddings
from model_training.data.config import OUTPUT_3DMM_PARAMS, OUTPUT_2D_LANDMARKS, OUTPUT_LANDMARKS_HEATMAP


logger = logging.getLogger(__name__)
_FILENAME = "dad_3dheads.trcd"
_PUBLIC_URL = "https://media.pinatafarm.com/public/research/dad-3dheads/dad_3dheads.trcd"


def model_exists() -> bool:
    return os.path.isfile(os.path.join(dirname, "../data", _FILENAME))


def download_model(url: str, retries: int = 5, verify_ssl: bool = True) -> None:
    """Download an given URL
        Parameters:
        ----------
        url : str
            URL to download
        retries : integer, default 5
            The number of times to attempt the download in case of failure or non 200 return codes
        verify_ssl : bool, default True
            Verify SSL certificates.
        """
    filename = os.path.join(dirname, "../data", _FILENAME)
    # os.path.join(os.path.expanduser("~"), ".dad_checkpoints", _FILENAME)
    assert retries >= 0, "Number of retries should be at least 0"

    if not verify_ssl:
        logger.warning(
            "Unverified HTTPS request is being made (verify_ssl=False). "
            "Adding certificate verification is strongly advised.")

    while retries + 1 > 0:
        try:
            logger.info("Downloading {} from {}...".format(filename, url))
            r = requests.get(url, stream=True, verify=verify_ssl)
            if r.status_code != 200:
                raise RuntimeError("Failed downloading url {}".format(url))
            with open(filename, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024):
                    if chunk:  # filter out keep-alive new chunks
                        f.write(chunk)
            break
        except Exception as e:
            retries -= 1
            if retries <= 0:
                raise e
            else:
                logger.info("download failed, retrying, {} attempt{} left".format(retries, "s" if retries > 1 else ""))

from torch.nn import functional as F
class FaceMeshPredictor:
    def __init__(self, config: Dict[str, Any], device: torch.device):
        self.device = device
        self.flame_constants = config["constants"]
        self.model = torch.jit.load(os.path.join(dirname, config["model_path"]))
        self.model = to_device(self.model, self.device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.head_mesh = HeadMesh(self.flame_constants).eval().requires_grad_(False)
        self._img_size = config["img_size"]
        self._stride = config.get("stride", 2)
        
        self.mean = torch.as_tensor([0.485, 0.456, 0.406], dtype=torch.float32, device=device).view(1, -1, 1, 1)
        self.std = torch.as_tensor([0.229, 0.224, 0.225], dtype=torch.float32, device=device).view(1, -1, 1, 1)

    def __call__(self, x: Any) -> Any:
        cache = {}
        x = self.preprocess(x, cache)
        res = self.process(x, cache)
        res = self.postprocess(res, cache)
        return res

    def preprocess(self, x: torch.Tensor, cache: Dict[str, Any], *kw: Any) -> Tensor:
        cache["input_shape"] = x.shape[-2:]
        x = F.interpolate(x, (self._img_size, self._img_size), align_corners=False, mode='bilinear', antialias=True)
        x = ((x / 2 + 0.5) - self.mean) / self.std
        return x

    def process(self, x: torch.Tensor, *kw: Any) -> Union[Tensor, Dict[str, Tensor]]:
        res = self.model(x)
        return res

    def _parse_output(self, x: Dict[str, torch.Tensor]) -> Union[Tuple[np.ndarray, np.ndarray], np.ndarray]:
        pred_3dmm = x[OUTPUT_3DMM_PARAMS]
        pred_3dmm = pred_3dmm

        if OUTPUT_2D_LANDMARKS in x.keys():
            pred_landmarks = x[OUTPUT_2D_LANDMARKS] * 256.0
        elif OUTPUT_LANDMARKS_HEATMAP in x.keys():
            pred_heatmap = x[OUTPUT_LANDMARKS_HEATMAP]
            # yx to xy
            pred_landmarks = unravel_index(torch.sigmoid(pred_heatmap)).flip(-1)[0]
            pred_landmarks = float(self._stride) * pred_landmarks
        else:
            return pred_3dmm
        return pred_landmarks, pred_3dmm
    
    def _get_paddings(self, cache: Dict[str, Any]) -> Tuple[List[int], float]:
        h, w = cache["input_shape"]
        max_side = max(h, w)
        scale = self._img_size / float(max_side)
        new_h, new_w = tuple(round(dim * scale) for dim in (h, w))
        paddings = calculate_paddings(new_h, new_w)
        return paddings, scale

    def _get_predictions(
            self, x: Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor], cache: Dict[str, Any]
    ) -> Dict[str, Any]:
        paddings, scale = self._get_paddings(cache)
        if type(x) is tuple:
            landmarks, pred_3dmm = x
            
            landmarks = landmarks.clamp(min=0, max=self._img_size)
            landmarks = self.readjust_landmarks_to_the_input_image(landmarks, paddings, scale)

            pred_3dmm = self.readjust_3dmm_to_the_input_image(pred_3dmm, paddings, scale)
            vertices_3d = self.head_mesh.vertices_3d(pred_3dmm)[0].squeeze()
            projected_vertices = self.head_mesh.reprojected_vertices(params_3dmm=pred_3dmm, to_2d=True)

            return {"points": landmarks,
                    "projected_vertices": projected_vertices,
                    "3d_vertices": vertices_3d,
                    "3dmm_params": pred_3dmm}

        pred_3dmm = self.readjust_3dmm_to_the_input_image(x, paddings, scale)
        return {"3dmm_params": pred_3dmm}

    def readjust_landmarks_to_the_input_image(
            self, landmarks: torch.Tensor, paddings: List[int], scale: float
    ) -> torch.Tensor:
        landmarks = landmarks - torch.as_tensor([[paddings[2], paddings[0]]], dtype=landmarks.dtype, device=landmarks.device)
        landmarks = (landmarks / scale)
        return landmarks

    def readjust_3dmm_to_the_input_image(
            self, pred_3dmm: torch.Tensor, paddings: List[int], scale: float
    ) -> torch.Tensor:
        scale_idx = self.find_3dmm_idx("scale", self.flame_constants)
        translation_idx = self.find_3dmm_idx("translation", self.flame_constants)

        old_flame_params_scale = pred_3dmm[:, scale_idx: scale_idx + self.flame_constants["scale"]]
        old_flame_params_translation = pred_3dmm[
                                       :, translation_idx: translation_idx + self.flame_constants["translation"]
                                       ]

        new_flame_params_scale = (old_flame_params_scale + 1.0) / scale - 1.0
        new_flame_params_translation = (
                                               old_flame_params_translation + 1.0 - torch.as_tensor(
                                           [[paddings[2], paddings[0], 0]], dtype=old_flame_params_translation.dtype, device=old_flame_params_translation.device) * 2 / self._img_size
                                       ) / scale - 1.0

        pred_3dmm[:, scale_idx: scale_idx + self.flame_constants["scale"]] = \
            new_flame_params_scale
        pred_3dmm[:, translation_idx: translation_idx + self.flame_constants["translation"]] = \
            new_flame_params_translation

        return pred_3dmm

    @staticmethod
    def find_3dmm_idx(key: str, consts: Dict[str, int]) -> int:
        idx = 0
        for k, v in consts.items():
            if k != key:
                idx += v
            else:
                break
        return idx

    def postprocess(self, x: Tuple[torch.Tensor, torch.Tensor], cache: Dict[str, Any], *kw: Any) -> Dict[str, Any]:
        output = self._parse_output(x)
        predictions = self._get_predictions(output, cache)
        if "points" in predictions.keys():
            predictions["points"] = torch.reshape(predictions["points"], (-1, 2))
        return predictions

    @classmethod
    def dad_3dnet(cls, device: torch.device):
        config = load_yaml(get_relative_path("dad_3dnet.yaml", __file__))
        if not model_exists():
            logger.info("Downloading the model")
            download_model(_PUBLIC_URL)
        return FaceMeshPredictor(config=config, device=device)

