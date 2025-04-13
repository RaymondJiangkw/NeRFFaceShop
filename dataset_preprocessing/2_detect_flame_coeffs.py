import os
import sys
sys.path.append(os.path.abspath(".."))

device = "cuda"

import numpy as np
np.bool = np.bool_
np.int = np.int_
np.float = np.float_
np.complex = np.complex_
np.object = np.object_
np.unicode = np.unicode_
np.str = np.str_

from external_dependencies.DAD3DHeads import DAD3DHeads
dad3dheads = DAD3DHeads(device)

import cv2
import torch
import argparse
fast_to_tensor = lambda x: (torch.from_numpy(
    cv2.cvtColor(cv2.imread(x), cv2.COLOR_BGR2RGB)).to(torch.float32) / 255. * 2 - 1).permute(2, 0, 1)[None, ...].to(device)

import pickle
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--out_dir", type=str, required=True)

if __name__ == "__main__":
    args = parser.parse_args()
    out_dir = args.out_dir
    image_out_dir = os.path.join(args.out_dir, 'images')
    flame_out_dir = os.path.join(args.out_dir, 'flames')
    assert os.path.exists(image_out_dir)
    os.makedirs(os.path.join(flame_out_dir), exist_ok=True)

    video_fn_s = sorted(os.listdir(image_out_dir))

    for video_fn in tqdm(video_fn_s):
        flame_code_s = {}
        fn_s = sorted(list(filter(lambda x: x.endswith('.jpg'), os.listdir(os.path.join(image_out_dir, video_fn)))))
        for fn in fn_s:
            codedict = dad3dheads(fast_to_tensor(os.path.join(image_out_dir, video_fn, fn)), 'flame')
            shape = codedict['shape'] # .squeeze().cpu().numpy().tolist()
            exp = codedict['expression'] # .squeeze().cpu().numpy().tolist()
            jaw = codedict['jaw'] # .squeeze().cpu().numpy().tolist()
            flame_code_s[fn] = {'shape': shape, 'exp': exp, 'jaw': jaw}
        with open(os.path.join(flame_out_dir, f'{video_fn}.pkl'), 'wb') as f:
            pickle.dump(flame_code_s, f)