import sys
sys.path.append("..")
import os
import argparse
import shutil
import pickle
import numpy as np
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--out_dir", type=str, required=True)

if __name__ == "__main__":
    args = parser.parse_args()
    out_dir = args.out_dir
    image_out_dir = os.path.join(args.out_dir, 'images')
    label_out_dir = os.path.join(args.out_dir, 'labels')
    flame_out_dir = os.path.join(args.out_dir, 'flames')
    assert os.path.exists(flame_out_dir)

    exp_std = {}
    jaw_std = {}
    shape_std = {}

    video_fn_s = sorted(os.listdir(flame_out_dir))

    for video_fn in tqdm(video_fn_s):
        v_fn = os.path.splitext(video_fn)[0]
        with open(os.path.join(flame_out_dir, video_fn), 'rb') as f:
            flame_code_s = pickle.load(f)
        shape_s = []
        exp_s = []
        jaw_s = []
        for fn in flame_code_s:
            shape_s.append(np.array(flame_code_s[fn]['shape']))
            exp_s.append(np.array(flame_code_s[fn]['exp']))
            jaw_s.append(np.array(flame_code_s[fn]['jaw']))
        shape_s = np.stack(shape_s)
        exp_s = np.stack(exp_s)
        jaw_s = np.stack(jaw_s)
        shape_std[v_fn] = shape_s.std(axis=0).sum()
        exp_std[v_fn] = exp_s.std(axis=0).sum()
        jaw_std[v_fn] = jaw_s.std(axis=0).sum()
    
    for video_fn in tqdm(video_fn_s):
        fn = os.path.splitext(video_fn)[0]
        if exp_std[fn] < 1.50 or jaw_std[fn] < 0.02:
            shutil.rmtree(os.path.join(image_out_dir, fn))
            os.remove(os.path.join(label_out_dir, fn + ".json"))
            os.remove(os.path.join(flame_out_dir, fn + ".pkl"))