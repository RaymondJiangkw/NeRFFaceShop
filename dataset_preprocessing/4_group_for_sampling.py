import sys
sys.path.append("..")
sys.path.append("../external_dependencies/")

import numpy as np
np.bool = np.bool_
np.int = np.int_
np.float = np.float_
np.complex = np.complex_
np.object = np.object_
np.unicode = np.unicode_
np.str = np.str_

import os
import numpy as np
import pickle
from tqdm import tqdm
import argparse
import torch
from external_dependencies.decalib.main import DECAWrapper

device = "cuda"
deca = DECAWrapper(device)

parser = argparse.ArgumentParser()
parser.add_argument("--out_dir", required=True, type=str)

if __name__ == "__main__":
    args = parser.parse_args()
    out_dir = args.out_dir
    image_out_dir = os.path.join(args.out_dir, 'images')
    label_out_dir = os.path.join(args.out_dir, 'labels')
    flame_out_dir = os.path.join(args.out_dir, 'flames')
    weight_out_dir = os.path.join(args.out_dir, 'weights')
    os.makedirs(weight_out_dir, exist_ok=True)

    video_fn_s = sorted(os.listdir(image_out_dir))

    threshold = 1.3

    for video_fn in tqdm(video_fn_s):
        with open(os.path.join(flame_out_dir, f'{video_fn}.pkl'), 'rb') as f:
            flame_code_s = pickle.load(f)
        
        fn_s = sorted(list(filter(lambda x: x.endswith('.jpg'), os.listdir(os.path.join(image_out_dir, video_fn)))))
        
        verts = []
        for fn in fn_s:
            exp = torch.as_tensor(flame_code_s[fn]["exp"])[None, :50].to(device)
            jaw = torch.as_tensor(flame_code_s[fn]["jaw"])[None, ...].to(device)
            verts.append(deca.deca.flame(
                shape_params=torch.zeros(1, 100).to(device), 
                expression_params=exp, pose_params=torch.cat((torch.zeros_like(jaw), jaw), dim=-1))[0])
        verts = torch.cat(verts)
        
        dists = np.stack([(torch.sum((verts[i:i+1, :] - verts[:, :]) ** 2, dim=-1) ** 0.5).sum(dim=-1).cpu().numpy() for i in range(len(fn_s))])
        del verts
        
        visited = np.array([0] * len(fn_s))
        groups = []
        
        for idx in range(len(fn_s)):
            if visited[idx] == 1:
                continue
            group_idxs = np.where(np.logical_and(dists[idx] < threshold, dists[idx] != 0))[0]
            group_idxs = np.concatenate((np.array([idx]), group_idxs))
            visited[group_idxs] = 1
            groups.append(group_idxs)
        
        probs = np.zeros(len(fn_s))
        for group in groups:
            probs[group] += 1. / len(groups) / len(group)
        with open(os.path.join(weight_out_dir, video_fn + '.npy'), 'wb') as f:
            np.save(f, probs)
            np.save(f, dists)
        del probs
        del dists