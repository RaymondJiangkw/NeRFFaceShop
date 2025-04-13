import sys
sys.path.append("..")
import os
import numpy as np
import pickle
from tqdm import tqdm
import argparse
import torch
import json
import cv2

fast_to_tensor = lambda x: (torch.from_numpy(
    cv2.cvtColor(cv2.imread(x), cv2.COLOR_BGR2RGB)).to(torch.float32) / 255. * 2 - 1).permute(2, 0, 1)[None, ...].to(device)

device = "cuda"

from external_dependencies.DPR import DPR
dpr = DPR.DPR(device)

parser = argparse.ArgumentParser()
parser.add_argument("--out_dir", required=True, type=str)

if __name__ == "__main__":
    args = parser.parse_args()
    out_dir = args.out_dir
    image_out_dir = os.path.join(args.out_dir, 'images')
    label_out_dir = os.path.join(args.out_dir, 'labels')
    flame_out_dir = os.path.join(args.out_dir, 'flames')
    weight_out_dir = os.path.join(args.out_dir, 'weights')
    sh_out_dir = os.path.join(args.out_dir, "shs")
    os.makedirs(sh_out_dir, exist_ok=True)

    video_fn_s = sorted(os.listdir(image_out_dir))

    for video_fn in tqdm(video_fn_s):
        fn_s = sorted(list(filter(lambda x: x.endswith('.jpg'), os.listdir(os.path.join(image_out_dir, video_fn)))))
        shs = {}
        for fn in fn_s:
            shs[fn] = dpr.extract_lighting(fast_to_tensor(os.path.join(image_out_dir, video_fn, fn))).squeeze().cpu().numpy().tolist()
        with open(os.path.join(sh_out_dir, f'{video_fn}.json'), 'w') as f:
            json.dump(shs, f)