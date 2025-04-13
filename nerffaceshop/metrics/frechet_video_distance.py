import os
import math
import uuid
import dnnlib
import hashlib
from torch_utils import misc

import copy
import torch
from torch.nn import functional as F

import scipy
import numpy as np
from .metric_utils import FeatureStats, get_feature_detector, get_feature_detector_name

#----------------------------------------------------------------------------

def preprocess(
    video: torch.Tensor,      # (T, C, H, W) in [0, 255]
    resolution: int = 224
):
    video = video.to(torch.float32) / 255.
    t, c, h, w = video.shape
    
    # scale shorter side to resolution
    scale = resolution / min(h, w)
    if h < w:
        target_size = (resolution, math.ceil(w * scale))
    else:
        target_size = (math.ceil(h * scale), resolution)
    video = F.interpolate(video, size=target_size, mode='bilinear', align_corners=False, antialias=True)

    # center crop
    t, c, h, w = video.shape
    w_start = (w - resolution) // 2
    h_start = (h - resolution) // 2
    video = video[:, :, h_start:h_start + resolution, w_start:w_start + resolution]
    video = video.permute(1, 0, 2, 3).contiguous() # (C, T, H, W)
    
    video = (video - 0.5) * 2
    
    return video # (C, T, H, W) in [-1, 1]

#----------------------------------------------------------------------------

def compute_feature_stats_for_dataset(opts, detector_url, detector_kwargs, rel_lo=0, rel_hi=1, data_loader_kwargs=None, max_items=None, **stats_kwargs):
    dataset = dnnlib.util.construct_class_by_name(**opts.dataset_kwargs)
    if data_loader_kwargs is None:
        data_loader_kwargs = dict(pin_memory=True, num_workers=3, prefetch_factor=2)
    
    # Try to lookup from cache.
    cache_file = None
    if opts.cache:
        # Choose cache file name.
        args = dict(dataset_kwargs=opts.dataset_kwargs, detector_url=detector_url, detector_kwargs=detector_kwargs, stats_kwargs=stats_kwargs, max_items=max_items)
        md5 = hashlib.md5(repr(sorted(args.items())).encode('utf-8'))
        cache_tag = f'{dataset.name}-{get_feature_detector_name(detector_url)}-{md5.hexdigest()}'
        cache_file = dnnlib.make_cache_dir_path('gan-metrics', cache_tag + '.pkl')
        
        # Check if the file exists (all processes must agree).
        flag = os.path.isfile(cache_file) if opts.rank == 0 else False
        if opts.num_gpus > 1:
            flag = torch.as_tensor(flag, dtype=torch.float32, device=opts.device)
            torch.distributed.broadcast(tensor=flag, src=0)
            flag = (float(flag.cpu()) != 0)

        # Load.
        if flag:
            return FeatureStats.load(cache_file)

    # Initialize.
    num_items = len(dataset)
    if max_items is not None:
        num_items = min(num_items, max_items)
    stats = FeatureStats(max_items=num_items, **stats_kwargs)
    progress = opts.progress.sub(tag='dataset features', num_items=num_items, rel_lo=rel_lo, rel_hi=rel_hi)
    detector = get_feature_detector(url=detector_url, device=opts.device, num_gpus=opts.num_gpus, rank=opts.rank, verbose=progress.verbose)

    # Main loop.
    item_subset = [(i * opts.num_gpus + opts.rank) % num_items for i in range((num_items - 1) // opts.num_gpus + 1)]
    for images, _labels in torch.utils.data.DataLoader(dataset=dataset, sampler=item_subset, batch_size=1, **data_loader_kwargs):
        assert len(images.shape) == 5 and images.shape[0] == 1
        if images.shape[2] == 1:
            images = images.repeat([1, 1, 3, 1, 1])
        images = preprocess(images[0])[None, ...]
        features = detector(images.to(opts.device), **detector_kwargs)
        stats.append_torch(features, num_gpus=opts.num_gpus, rank=opts.rank)
        progress.update(stats.num_items)

    # Save to cache.
    if cache_file is not None and opts.rank == 0:
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        temp_file = cache_file + '.' + uuid.uuid4().hex
        stats.save(temp_file)
        os.replace(temp_file, cache_file) # atomic
    return stats

#----------------------------------------------------------------------------

def compute_feature_stats_for_generator(opts, detector_url, detector_kwargs, rel_lo=0, rel_hi=1, batch_gen=12, data_loader_kwargs=None, **stats_kwargs):
    assert batch_gen > 0
    
    # Setup generator and labels.
    G = copy.deepcopy(opts.G).eval().requires_grad_(False).to(opts.device)
    E = opts.E.eval().requires_grad_(False).to(opts.device) if opts.E is not None else None
    
    dataset = dnnlib.util.construct_class_by_name(**opts.dataset_kwargs)
    if data_loader_kwargs is None:
        data_loader_kwargs = dict(pin_memory=True, num_workers=3, prefetch_factor=2)

    # Initialize.
    stats = FeatureStats(**stats_kwargs)
    assert stats.max_items is not None
    progress = opts.progress.sub(tag='generator features', num_items=stats.max_items, rel_lo=rel_lo, rel_hi=rel_hi)
    detector = get_feature_detector(url=detector_url, device=opts.device, num_gpus=opts.num_gpus, rank=opts.rank, verbose=progress.verbose)
    
    sampler = misc.InfiniteSampler(dataset=dataset, rank=opts.rank, num_replicas=opts.num_gpus, shuffle=False)

    # Main loop.
    for _images, _labels in torch.utils.data.DataLoader(dataset=dataset, sampler=sampler, batch_size=1, **data_loader_kwargs):
        _images, _labels = _images[0], _labels[0]
        if stats.is_full():
            break
        frame_size = len(_images)
        _images = _images.to(torch.float32) / 127.5 - 1
        images = []
        
        z = torch.randn([1, G.z_dim], device=opts.device)
        c = _labels[0:1].to(opts.device) # Following the PV3D, using the camera parameter of frame 0 as the condition.
        w = G.backbone.mapping(z, c)
        for _i in range(frame_size // batch_gen):
            c = _labels[_i*batch_gen:(_i+1)*batch_gen].to(opts.device)
            if E is None:
                d = torch.randn([c.size(0), G.d_dim], device=opts.device)
                wd = G.backbone.dmapping(d, None)
            else:
                wd = E(F.interpolate(_images[_i*batch_gen:(_i+1)*batch_gen].to(opts.device), size=(256, 256), mode='bilinear', align_corners=False, antialias=True))
            imgs = G.synthesis(w.expand(wd.size(0), -1, -1), wd, c, noise_mode='const', **opts.synthesis_kwargs)['image'] # Again, following the PV3D to set the `noise_mode`.
            imgs = (imgs * 127.5 + 128).clamp(0, 255).to(torch.uint8)
            images.append(imgs.cpu())
        images = torch.cat(images)
        if images.shape[1] == 1:
            images = images.repeat([1, 3, 1, 1])
        images = preprocess(images)[None, ...]
        features = detector(images.to(opts.device), **detector_kwargs)
        stats.append_torch(features, num_gpus=opts.num_gpus, rank=opts.rank)
        progress.update(stats.num_items)
    return stats

#----------------------------------------------------------------------------

def compute_fvd(opts, max_real, num_gen):
    detector_url = 'i3d'
    detector_kwargs = dict()

    mu_real, sigma_real = compute_feature_stats_for_dataset(
        opts=opts, detector_url=detector_url, detector_kwargs=detector_kwargs,
        rel_lo=0, rel_hi=0, capture_mean_cov=True, max_items=max_real).get_mean_cov()

    mu_gen, sigma_gen = compute_feature_stats_for_generator(
        opts=opts, detector_url=detector_url, detector_kwargs=detector_kwargs,
        rel_lo=0, rel_hi=1, capture_mean_cov=True, max_items=num_gen).get_mean_cov()

    if opts.rank != 0:
        return float('nan')

    m = np.square(mu_gen - mu_real).sum()
    s, _ = scipy.linalg.sqrtm(np.dot(sigma_gen, sigma_real), disp=False) # pylint: disable=no-member
    fid = np.real(m + np.trace(sigma_gen + sigma_real - s * 2))
    return float(fid)

#----------------------------------------------------------------------------