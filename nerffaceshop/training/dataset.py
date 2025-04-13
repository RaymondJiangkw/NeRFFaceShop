# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Streaming images and labels from datasets created with dataset_tool.py."""

import os
import numpy as np
import zipfile
import PIL.Image
import json
import torch
import dnnlib
import random as r
try:
    import pyspng
except ImportError:
    pyspng = None

#----------------------------------------------------------------------------

class Dataset(torch.utils.data.Dataset):
    def __init__(self,
        name,                   # Name of the dataset.
        raw_shape,              # Shape of the raw image data (NCHW).
        max_size    = None,     # Artificially limit the size of the dataset. None = no limit. Applied before xflip.
        use_labels  = False,    # Enable conditioning labels? False = label dimension is zero.
        xflip       = False,    # Artificially double the size of the dataset via x-flips. Applied after max_size.
        random_seed = 0,        # Random seed to use when applying max_size.
    ):
        self._name = name
        self._raw_shape = list(raw_shape)
        self._use_labels = use_labels
        self._raw_labels = None
        self._label_shape = None

        # Apply max_size.
        self._raw_idx = np.arange(self._raw_shape[0], dtype=np.int64)
        if (max_size is not None) and (self._raw_idx.size > max_size):
            np.random.RandomState(random_seed).shuffle(self._raw_idx)
            self._raw_idx = np.sort(self._raw_idx[:max_size])

        # Apply xflip.
        self._xflip = np.zeros(self._raw_idx.size, dtype=np.uint8)
        if xflip:
            self._raw_idx = np.tile(self._raw_idx, 2)
            self._xflip = np.concatenate([self._xflip, np.ones_like(self._xflip)])
        self._raw_labels_std = None

    def _get_raw_labels(self):
        if self._raw_labels is None:
            self._raw_labels = self._load_raw_labels() if self._use_labels else None
            if self._raw_labels is None:
                self._raw_labels = np.zeros([self._raw_shape[0], 0], dtype=np.float32)
            if isinstance(self._raw_labels, np.ndarray):
                self._raw_labels = (self._raw_labels, )
            assert isinstance(self._raw_labels, tuple)
            for _raw_label in self._raw_labels:
                assert isinstance(_raw_label, np.ndarray)
                assert _raw_label.shape[0] == self._raw_shape[0]
                assert _raw_label.dtype in [np.float32, np.int64]
                if _raw_label.dtype == np.int64:
                    assert _raw_label.ndim == 1
                    assert np.all(_raw_label >= 0)
            self._raw_labels_std = [np.concatenate(l).std(0) for l in self._raw_labels]
        return self._raw_labels

    def close(self): # to be overridden by subclass
        pass

    def _load_raw_image(self, raw_idx): # to be overridden by subclass
        raise NotImplementedError

    def _load_raw_labels(self): # to be overridden by subclass
        raise NotImplementedError

    def __getstate__(self):
        return dict(self.__dict__, _raw_labels=None)

    def __del__(self):
        try:
            self.close()
        except:
            pass

    def __len__(self):
        return self._raw_idx.size

    def __getitem__(self, idx):
        image = self._load_raw_image(self._raw_idx[idx])
        assert isinstance(image, np.ndarray)
        assert list(image.shape) == self.image_shape
        assert image.dtype == np.uint8
        if self._xflip[idx]:
            assert image.ndim == 3 # CHW
            image = image[:, :, ::-1]
        labels = self.get_label(idx)
        return image.copy(), *labels

    def get_label(self, idx):
        labels = [_raw_labels[self._raw_idx[idx]] for _raw_labels in self._get_raw_labels()]
        out = []
        for label in labels:
            if label.dtype == np.int64:
                onehot = np.zeros(self.label_shape, dtype=np.float32)
                onehot[label] = 1
                label = onehot
            out.append(label.copy())
        return out

    def get_details(self, idx):
        d = dnnlib.EasyDict()
        d.raw_idx = int(self._raw_idx[idx])
        d.xflip = (int(self._xflip[idx]) != 0)
        d.raw_label = [_raw_labels[d.raw_idx].copy() for _raw_labels in self._get_raw_labels()]
        return d

    def get_label_std(self):
        if self._raw_labels_std is None:
            self._get_raw_labels()
        return self._raw_labels_std

    @property
    def name(self):
        return self._name

    @property
    def image_shape(self):
        return list(self._raw_shape[1:])

    @property
    def num_channels(self):
        assert len(self.image_shape) == 3 # CHW
        return self.image_shape[0]

    @property
    def resolution(self):
        assert len(self.image_shape) == 3 # CHW
        assert self.image_shape[1] == self.image_shape[2]
        return self.image_shape[1]

    @property
    def label_shape(self):
        if self._label_shape is None:
            raw_labels = self._get_raw_labels()
            self._label_shape = []
            for raw_label in raw_labels:
                if raw_label.dtype == np.int64:
                    self._label_shape.append(int(np.max(raw_label)) + 1)
                else:
                    assert len(raw_label.shape) == 2
                    self._label_shape.append(raw_label.shape[1])
        return self._label_shape

    @property
    def label_dim(self):
        # assert len(self.label_shape) == 1
        return self.label_shape

    @property
    def has_labels(self):
        return any(x != 0 for x in self.label_shape)

    @property
    def has_onehot_labels(self):
        for _raw_label in self._get_raw_labels():
            if _raw_label.dtype == np.int64:
                return True
        return False

#----------------------------------------------------------------------------

def flip_yaw(pose_matrix):
    flipped = pose_matrix
    flipped[0, 1] *= -1
    flipped[0, 2] *= -1
    flipped[1, 0] *= -1
    flipped[2, 0] *= -1
    flipped[0, 3] *= -1
    return flipped

def flip(args):
    image, cam, sh = args
    image = image[:, :, ::-1].copy()
    cam = cam.reshape(-1).copy()
    sh = sh.reshape(-1).copy()
    
    pose, intrinsics = cam[:16].reshape(4, 4), cam[16:].reshape(3, 3)
    flipped_pose = flip_yaw(pose)
    cam = np.concatenate([flipped_pose.reshape(-1), intrinsics.reshape(-1)])
    
    sh[[3, 4, 7]] = -sh[[3, 4, 7]]
    
    return [image, cam, sh]

import random as r

class VideoFolderDataset(Dataset):
    def __init__(self,
        path,                   # Path to directory or zip.
        resolution      = None, # Ensure specific resolution, None = highest available.
        sampling_strategy = 'adaptive', # Frame Sampling Strategy: 'adaptive', 'uniform'.
        deterministic = False,  # Whether it is deterministic when sampling.
        **super_kwargs,         # Additional arguments for the Dataset base class.
    ):
        self._path = path
        self.sampling_strategy = sampling_strategy
        self.deterministic = deterministic
        assert os.path.isdir(self._path)
        assert sampling_strategy in ['adaptive', 'uniform']

        self._image_path = os.path.join(self._path, "images")
        self._camera_path = os.path.join(self._path, "labels")
        self._sh_path = os.path.join(self._path, "shs")
        self._weight_path = os.path.join(self._path, "weights")
        
        PIL.Image.init()
        self._video_fnames = sorted(os.listdir(self._image_path))
        if len(self._video_fnames) == 0:
            raise IOError('No video files found in the specified path')
        self._video_image_fns = [None for _i in range(len(self._video_fnames))]
        self._video_weights = None
        self._video_dists = None

        name = os.path.splitext(os.path.basename(self._path))[0]
        raw_shape = [len(self._video_fnames)] + list(self._load_raw_image(0, 0).shape)
        if resolution is not None and (raw_shape[2] != resolution or raw_shape[3] != resolution):
            raise IOError('Image files do not match the specified resolution')
        super().__init__(name=name, raw_shape=raw_shape, **super_kwargs)

    @staticmethod
    def _file_ext(fname):
        return os.path.splitext(fname)[1].lower()

    def _open_file(self, fname):
        return open(os.path.join(self._image_path, fname), 'rb')

    def close(self):
        pass

    def __getstate__(self):
        return dict(super().__getstate__(), _zipfile=None)
    
    def _load_video_image_fns(self, raw_idx):
        return sorted(list(filter(lambda x: x.endswith('.jpg') or x.endswith('.png'), os.listdir(os.path.join(self._image_path, self._video_fnames[raw_idx])))))
    
    def _get_video_length(self, raw_idx):
        if self._video_image_fns[raw_idx] is None:
            self._video_image_fns[raw_idx] = self._load_video_image_fns(raw_idx)
        return len(self._video_image_fns[raw_idx])
    
    def _get_video_image_fn(self, raw_idx, frame_idx):
        if self._video_image_fns[raw_idx] is None:
            self._video_image_fns[raw_idx] = self._load_video_image_fns(raw_idx)
        return self._video_image_fns[raw_idx][frame_idx]
    
    def _load_raw_image(self, raw_idx, frame_idx):
        vname = self._video_fnames[raw_idx]
        fname = self._get_video_image_fn(raw_idx, frame_idx)
        with self._open_file(os.path.join(vname, fname)) as f:
            image = np.array(PIL.Image.open(f))
            if image.shape[0] == 563: # Account for PanoHead Expansion
                image = image[51:, 25:-26, :]
        if image.ndim == 2:
            image = image[:, :, np.newaxis] # HW => HWC
        image = image.transpose(2, 0, 1) # HWC => CHW
        return image

    def _load_raw_labels(self):
        cams = {}
        shs = {}
        for fname in self._video_fnames:
            with open(os.path.join(self._camera_path, fname + '.json'), 'r') as f:
                cams[fname] = json.load(f) # (F, 4*4+3*3)
            with open(os.path.join(self._sh_path, fname + '.json'), 'r') as f:
                shs[fname] = json.load(f) # (F, 9)
        return cams, shs
    
    def _get_raw_labels(self):
        if self._raw_labels is None:
            self._raw_labels = self._load_raw_labels() if self._use_labels else None
            assert self._raw_labels is not None
            assert isinstance(self._raw_labels, tuple)
            self._raw_labels_std = [np.concatenate([np.array(list(l[vn].values())) for vn in l]).std(0) for l in self._raw_labels]
        return self._raw_labels
    
    def _load_video_weights(self):
        weights = []
        dists = []
        for fname in self._video_fnames:
            with open(os.path.join(self._weight_path, fname + '.npy'), 'rb') as f:
                weights.append(np.load(f))
                dists.append(np.load(f))
        return weights, dists
    
    def _get_video_weights(self):
        if self._video_weights is None or self._video_dists is None:
            self._video_weights, self._video_dists = self._load_video_weights()
        return self._video_weights, self._video_dists
    
    def _get_video_weights_by_name(self, fname):
        with open(os.path.join(self._weight_path, fname + '.npy'), 'rb') as f:
            weights = np.load(f)
            dists = np.load(f)
        return weights, dists
    
    def sample_frames(self, idx):
        if not self.deterministic:
            if self.sampling_strategy == 'adaptive':
                video_weights, video_dists = self._get_video_weights_by_name(self._video_fnames[self._raw_idx[idx]])
                # list(map(lambda pair: pair[self._raw_idx[idx]], self._get_video_weights()))
                assert self._get_video_length(self._raw_idx[idx]) == len(video_weights), f'{self._video_fnames[self._raw_idx[idx]]}, {self._get_video_length(self._raw_idx[idx])}, {len(video_weights)}'
                start_idx = r.choices(range(self._get_video_length(self._raw_idx[idx])), video_weights)[0]
                diffs = video_dists[start_idx] ** 0.5
                diffs = diffs / (diffs.sum() + 1e-5)
                if diffs.sum() > 0:
                    stop_idx = r.choices(range(self._get_video_length(self._raw_idx[idx])), diffs)[0]
                else:
                    print(self._video_fnames[self._raw_idx[idx]], start_idx, video_dists[start_idx])
                    stop_idx = r.choice(range(self._get_video_length(self._raw_idx[idx])))
            elif self.sampling_strategy == 'uniform':
                start_idx = r.choice(range(self._get_video_length(self._raw_idx[idx])))
                stop_idx = r.choice(range(self._get_video_length(self._raw_idx[idx])))
            return min(start_idx, stop_idx), max(start_idx, stop_idx)
        else:
            return 0, 1

    def __getitem__(self, idx):
        is_flip = r.random() > 0.5
        return sum([ (flip if is_flip else lambda x: x)(self.get_image(idx, frame_idx) + self.get_label(idx, frame_idx)) for frame_idx in self.sample_frames(idx) ], [])

    def get_label(self, idx, frame_idx):
        vname = self._video_fnames[self._raw_idx[idx]]
        fname = self._get_video_image_fn(idx, frame_idx)
        return list(map(lambda label: np.array(label[vname][fname].copy()), self._get_raw_labels()))
    
    def get_image(self, idx, frame_idx):
        image = self._load_raw_image(self._raw_idx[idx], frame_idx)
        assert isinstance(image, np.ndarray)
        assert list(image.shape) == self.image_shape
        assert image.dtype == np.uint8
        return [image.copy()]
    
    def get_details(self, idx):
        d = dnnlib.EasyDict()
        d.raw_idx = int(self._raw_idx[idx])
        d.xflip = (int(self._xflip[idx]) != 0)
        # d.raw_label = list(map(lambda label: label[d.raw_idx].copy(), self._get_raw_labels()))
        return d
    
    @property
    def label_shape(self):
        if self._label_shape is None:
            raw_labels = self._get_raw_labels()
            self._label_shape = []
            for raw_label in raw_labels:
                raw_label = np.array(list(list(raw_label.values())[0].values())[0])
                if raw_label.dtype == np.int64:
                    self._label_shape.append(int(np.max(raw_label)) + 1)
                else:
                    self._label_shape.append(raw_label.shape[0])
        return self._label_shape

    @property
    def has_labels(self):
        return True

    @property
    def has_onehot_labels(self):
        return False
    
#----------------------------------------------------------------------------

class ImageFolderDataset(Dataset):
    def __init__(self,
        path,                   # Path to directory or zip.
        resolution      = None, # Ensure specific resolution, None = highest available.
        **super_kwargs,         # Additional arguments for the Dataset base class.
    ):
        self._path = path
        self._resolution = resolution
        self._zipfile = None

        if os.path.isdir(self._path):
            self._type = 'dir'
            self._all_fnames = {os.path.relpath(os.path.join(root, fname), start=self._path) for root, _dirs, files in os.walk(self._path) for fname in files}
        elif self._file_ext(self._path) == '.zip':
            self._type = 'zip'
            self._all_fnames = set(self._get_zipfile().namelist())
        else:
            raise IOError('Path must point to a directory or zip')

        PIL.Image.init()
        self._image_fnames = sorted(fname for fname in self._all_fnames if self._file_ext(fname) in PIL.Image.EXTENSION)
        if len(self._image_fnames) == 0:
            raise IOError('No image files found in the specified path')

        name = os.path.splitext(os.path.basename(self._path))[0]
        raw_shape = [len(self._image_fnames)] + list(self._load_raw_image(0).shape)
        if resolution is not None and (raw_shape[2] != resolution or raw_shape[3] != resolution):
            raise IOError('Image files do not match the specified resolution')
        super().__init__(name=name, raw_shape=raw_shape, **super_kwargs)

    @staticmethod
    def _file_ext(fname):
        return os.path.splitext(fname)[1].lower()

    def _get_zipfile(self):
        assert self._type == 'zip'
        if self._zipfile is None:
            self._zipfile = zipfile.ZipFile(self._path)
        return self._zipfile

    def _open_file(self, fname):
        if self._type == 'dir':
            return open(os.path.join(self._path, fname), 'rb')
        if self._type == 'zip':
            return self._get_zipfile().open(fname, 'r')
        return None

    def close(self):
        try:
            if self._zipfile is not None:
                self._zipfile.close()
        finally:
            self._zipfile = None

    def __getstate__(self):
        return dict(super().__getstate__(), _zipfile=None)

    def _load_raw_image(self, raw_idx):
        fname = self._image_fnames[raw_idx]
        with self._open_file(fname) as f:
            if pyspng is not None and self._file_ext(fname) == '.png':
                image = pyspng.load(f.read())
            else:
                image = np.array(PIL.Image.open(f)) # .resize((self._resolution, self._resolution), PIL.Image.ANTIALIAS)
        if image.ndim == 2:
            image = image[:, :, np.newaxis] # HW => HWC
        image = image.transpose(2, 0, 1) # HWC => CHW
        return image
    
    def _get_raw_labels(self):
        if self._raw_labels is None:
            self._raw_labels = self._load_raw_labels() if self._use_labels else None
            assert self._raw_labels is not None
            assert isinstance(self._raw_labels, tuple)
            self._raw_labels_std = [l.std(0) for l in self._raw_labels]
        return self._raw_labels

    def _load_raw_labels(self):
        fname = 'dataset.json'
        if fname not in self._all_fnames:
            return None
        with self._open_file(fname) as f:
            data = json.load(f)
            labels = data['labels']
            shs = data['sh']
        if labels is None:
            return None
        labels = dict(labels)
        labels = [labels[os.path.basename(fname).replace('\\', '/')] for fname in self._image_fnames]
        labels = np.array(labels)
        labels = labels.astype({1: np.int64, 2: np.float32}[labels.ndim])
        
        shs = dict(shs)
        shs = [shs[os.path.basename(fname).replace('\\', '/')] for fname in self._image_fnames]
        shs = np.array(shs)
        shs = shs.astype({1: np.int64, 2: np.float32}[shs.ndim])
        
        return labels, shs

#----------------------------------------------------------------------------