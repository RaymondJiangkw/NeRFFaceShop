import sys
sys.path.append("..")
import cv2
import dlib
import numpy as np
import os
import json
import imageio
import yaml
from external_dependencies.FaceBoxes import FaceBoxes
from external_dependencies.TDDFA import TDDFA
from external_dependencies.TDDFA.utils.pose import P2sRt, matrix2angle
from tqdm import tqdm
import mediapipe as mp

debug = False

base_dir = os.path.dirname(__file__)

# load face detector
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(os.path.join(base_dir, "../external_dependencies/data/shape_predictor_68_face_landmarks.dat"))
cfg = yaml.load(open(os.path.join(base_dir, '../external_dependencies/TDDFA/configs/mb1_120x120.yml')), Loader=yaml.SafeLoader)
cfg['checkpoint_fp'] = os.path.join(base_dir, '../external_dependencies/TDDFA/weights/mb1_120x120.pth')
# print(cfg['checkpoint_fp'])
gpu_mode = True
tddfa = TDDFA(gpu_mode=gpu_mode, **cfg)
face_boxes = FaceBoxes()

hands = mp.solutions.hands.Hands(
    static_image_mode=True,
    max_num_hands=2,
    min_detection_confidence=0.5
)

SIZE = 512
MIN_SIZE = 512 if not debug else 256

def is_contain_hand(img):
    results = hands.process(img)
    return results.multi_hand_landmarks is not None

def eg3dcamparams(R_in):
    camera_dist = 2.7
    intrinsics = np.array([[4.2647, 0, 0.5], [0, 4.2647, 0.5], [0, 0, 1]])
    # assume inputs are rotation matrices for world2cam projection
    R = np.array(R_in).astype(np.float32).reshape(4,4)
    # add camera translation
    t = np.eye(4, dtype=np.float32)
    t[2, 3] = - camera_dist

    # convert to OpenCV camera
    convert = np.array([
        [1, 0, 0, 0],
        [0, -1, 0, 0],
        [0, 0, -1, 0],
        [0, 0, 0, 1],
    ]).astype(np.float32)

    # world2cam -> cam2world
    P = convert @ t @ R
    cam2world = np.linalg.inv(P)

    # add intrinsics
    label_new = np.concatenate([cam2world.reshape(16), intrinsics.reshape(9)], -1).tolist()
    return label_new

def find_center_bbox(roi_box_lst, w, h):
    bboxes = np.array(roi_box_lst)
    dx = 0.5*(bboxes[:,0] + bboxes[:,2]) - 0.5*(w-1)
    dy = 0.5*(bboxes[:,1] + bboxes[:,3]) - 0.5*(h-1)
    dist = np.stack([dx,dy],1)
    return np.argmin(np.linalg.norm(dist, axis=1))

def crop_final(
    img, 
    size=512, 
    quad=None,
    top_expand=0., 
    left_expand=0., 
    bottom_expand=0., 
    right_expand=0., 
    blur_kernel=None,
    borderMode=cv2.BORDER_REFLECT,
    upsample=2,
    min_size=MIN_SIZE,
):  

    orig_size = min(np.linalg.norm(quad[1] - quad[0]), np.linalg.norm(quad[2] - quad[1]))
    if min_size is not None and orig_size < min_size:
        assert False

    crop_w = int(size * (1 + left_expand + right_expand))
    crop_h = int(size * (1 + top_expand + bottom_expand))
    crop_size = (crop_w, crop_h)
    
    top = int(size * top_expand)
    left = int(size * left_expand)
    size -= 1
    bound = np.array([[left, top], [left, top + size], [left + size, top + size], [left + size, top]],
                        dtype=np.float32)

    mat = cv2.getAffineTransform(quad[:3], bound[:3])
    if upsample is None or upsample == 1:
        crop_img = cv2.warpAffine(np.array(img), mat, crop_size, flags=cv2.INTER_LANCZOS4, borderMode=borderMode)
    else:
        assert isinstance(upsample, int)
        crop_size_large = (crop_w*upsample,crop_h*upsample)
        crop_img = cv2.warpAffine(np.array(img), upsample*mat, crop_size_large, flags=cv2.INTER_LANCZOS4, borderMode=borderMode)
        crop_img = cv2.resize(crop_img, crop_size, interpolation=cv2.INTER_AREA) 

    empty = np.ones_like(img) * 255
    crop_mask = cv2.warpAffine(empty, mat, crop_size)

    if True:
        mask_kernel = int(size*0.02)*2+1
        blur_kernel = int(size*0.03)*2+1 if blur_kernel is None else blur_kernel
        downsample_size = (crop_w//8, crop_h//8)
        
        if crop_mask.mean() < 255:
            blur_mask = cv2.blur(crop_mask.astype(np.float32).mean(2),(mask_kernel,mask_kernel)) / 255.0
            blur_mask = blur_mask[...,np.newaxis]#.astype(np.float32) / 255.0
            blurred_img = cv2.blur(crop_img, (blur_kernel, blur_kernel), 0)
            crop_img = crop_img * blur_mask + blurred_img * (1 - blur_mask)
            crop_img = crop_img.astype(np.uint8)
    
    return crop_img

def get_crop_bound(lm, method="ffhq"):
    if len(lm) == 106:
        left_e = lm[104]
        right_e = lm[105]
        nose = lm[49]
        left_m = lm[84]
        right_m = lm[90]
        center = (lm[1] + lm[31]) * 0.5
    elif len(lm) == 68:
        left_e = np.mean(lm[36:42], axis=0)
        right_e = np.mean(lm[42:48], axis=0)
        nose = lm[33]
        left_m = lm[48]
        right_m = lm[54]
        center = (lm[0] + lm[16]) * 0.5
    else:
        raise ValueError(f"Unknown type of keypoints with a length of {len(lm)}")

    if method == "ffhq":
        eye_to_eye = right_e - left_e
        eye_avg = (left_e + right_e) * 0.5
        mouth_avg = (left_m + right_m) * 0.5
        eye_to_mouth = mouth_avg - eye_avg
        x = eye_to_eye - np.flipud(eye_to_mouth) * [-1, 1]
        x /= np.hypot(*x)
        x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)
        y = np.flipud(x) * [-1, 1]
        c = eye_avg + eye_to_mouth * 0.1
    elif method == "default":
        eye_to_eye = right_e - left_e
        eye_avg = (left_e + right_e) * 0.5
        eye_to_nose = nose - eye_avg
        x = eye_to_eye.copy()
        x /= np.hypot(*x)
        x *= max(np.hypot(*eye_to_eye) * 2.4, np.hypot(*eye_to_nose) * 2.75)
        y = np.flipud(x) * [-1, 1]
        c = center
    else:
        raise ValueError('%s crop method not supported yet.' % method)
    quad = np.stack([c - x - y, c - x + y, c + x + y, c + x - y])
    return quad.astype(np.float32), c, x, y

def crop_image(img, mat, crop_w, crop_h, upsample=1, borderMode=cv2.BORDER_CONSTANT):
    crop_size = (crop_w, crop_h)
    if upsample is None or upsample == 1:
        crop_img = cv2.warpAffine(np.array(img), mat, crop_size, flags=cv2.INTER_LANCZOS4, borderMode=borderMode)
    else:
        assert isinstance(upsample, int)
        crop_size_large = (crop_w*upsample,crop_h*upsample)
        crop_img = cv2.warpAffine(np.array(img), upsample*mat, crop_size_large, flags=cv2.INTER_LANCZOS4, borderMode=borderMode)
        crop_img = cv2.resize(crop_img, crop_size, interpolation=cv2.INTER_AREA) 
    return crop_img

def process_video(path_to_video: str, image_out_dir: str, label_out_dir: str):
    video_name = os.path.basename(path_to_video).split(".")[0]
    label_file_name = os.path.join(label_out_dir, video_name + '.json')
    if os.path.exists(label_file_name):
        return
    image_folder_name = os.path.join(image_out_dir, video_name)

    try:
        reader = imageio.get_reader(path_to_video)
    except:
        reader = None
    
    labels = {}
    if reader is not None:
        for frame_idx, frame in enumerate(reader):
            if debug and frame_idx > 10: break
            assert frame.shape[-1] == 3 and frame.dtype == np.uint8
        # for frame_idx in tqdm(range(len(frames))):
            # frame = frames[frame_idx]
            if is_contain_hand(frame): continue
            # gray scale    for frame_idx, frame in enumerate(reader):
            assert frame.shape[-1] == 3 and frame.dtype == np.uint8
        # for frame_idx in tqdm(range(len(frames))):
            # frame = frames[frame_idx]
            if is_contain_hand(frame): continue
            # gray scale
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            # detect face
            rects = detector(gray, 1)
            if len(rects) <= 0:
                continue
            # select the largest rect
            rect = max(rects, key=lambda x: abs((x.right() - x.left()) * (x.top() - x.bottom())))
            # get keypoints
            shape = predictor(gray, rect)
            # save kps to the dict
            landmark = [np.array([p.x, p.y]) for p in shape.parts()]
            # get quad
            quad, quad_c, quad_x, quad_y = get_crop_bound(landmark)
            # init crop for detecting face
            bound = np.array([[0, 0], [0, SIZE-1], [SIZE-1, SIZE-1], [SIZE-1, 0]], dtype=np.float32)
            mat = cv2.getAffineTransform(quad[:3], bound[:3])
            img = crop_image(frame, mat, SIZE, SIZE)
            h, w = img.shape[:2]

            boxes = face_boxes(img)
            
            if len(boxes) == 0:
                continue

            param_lst, roi_box_lst = tddfa(img, boxes)
            box_idx = find_center_bbox(roi_box_lst, w, h)


            param = param_lst[box_idx]
            P = param[:12].reshape(3, -1)  # camera matrix
            s_relative, R, t3d = P2sRt(P)
            pose = matrix2angle(R)
            pose = [p * 180 / np.pi for p in pose]

            # Adjust z-translation in object space
            R_ = param[:12].reshape(3, -1)[:, :3]
            u = tddfa.bfm.u.reshape(3, -1, order='F')
            trans_z = np.array([ 0, 0, 0.5*u[2].mean() ]) # Adjust the object center
            trans = np.matmul(R_, trans_z.reshape(3,1))
            t3d += trans.reshape(3)

            ''' Camera extrinsic estimation for GAN training '''
            # Normalize P to fit in the original image (before 3DDFA cropping)
            sx, sy, ex, ey = roi_box_lst[0]
            scale_x = (ex - sx) / tddfa.size
            scale_y = (ey - sy) / tddfa.size
            t3d[0] = (t3d[0]-1) * scale_x + sx
            t3d[1] = (tddfa.size-t3d[1]) * scale_y + sy
            t3d[0] = (t3d[0] - 0.5*(w-1)) / (0.5*(w-1)) # Normalize to [-1,1]
            t3d[1] = (t3d[1] - 0.5*(h-1)) / (0.5*(h-1)) # Normalize to [-1,1], y is flipped for image space
            t3d[1] *= -1
            t3d[2] = 0 # orthogonal camera is agnostic to Z (the model always outputs 66.67)

            s_relative = s_relative * 2000
            scale_x = (ex - sx) / (w-1)
            scale_y = (ey - sy) / (h-1)
            s = (scale_x + scale_y) / 2 * s_relative
            # print(f"[{iteration}] s={s} t3d={t3d}")

            if s < 0.7 or s > 1.3:
                continue
            if abs(pose[0]) > 90 or abs(pose[1]) > 80 or abs(pose[2]) > 50:
                continue
            if abs(t3d[0]) > 1. or abs(t3d[1]) > 1.:
                continue

            quad_c = quad_c + quad_x * t3d[0]
            quad_c = quad_c - quad_y * t3d[1]
            quad_x = quad_x * s
            quad_y = quad_y * s
            c, x, y = quad_c, quad_x, quad_y
            quad = np.stack([c - x - y, c - x + y, c + x + y, c + x - y]).astype(np.float32)
            if quad[0][0] < -32 or quad[0][1] < -32:
                continue
            orig_size = min(np.linalg.norm(quad[1] - quad[0]), np.linalg.norm(quad[2] - quad[1]))
            if orig_size < MIN_SIZE:
                continue

            s = 1
            t3d = 0 * t3d
            R[:,:3] = R[:,:3] * s
            P = np.concatenate([R,t3d[:,None]],1)
            P = np.concatenate([P, np.array([[0,0,0,1.]])], 0)

            pose = eg3dcamparams(P.flatten())
            cropped_img = crop_final(frame, size=SIZE, quad=quad)

            fn = str(frame_idx).zfill(5) + '.jpg'

            # print(frame_idx, os.path.join(image_folder_name, fn))

            labels[fn] = pose
            os.makedirs(image_folder_name, exist_ok=True)
            cv2.imwrite(os.path.join(image_folder_name, fn), cv2.cvtColor(cropped_img, cv2.COLOR_RGB2BGR))
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            # detect face
            rects = detector(gray, 1)
            if len(rects) <= 0:
                continue
            # select the largest rect
            rect = max(rects, key=lambda x: abs((x.right() - x.left()) * (x.top() - x.bottom())))
            # get keypoints
            shape = predictor(gray, rect)
            # save kps to the dict
            landmark = [np.array([p.x, p.y]) for p in shape.parts()]
            # get quad
            quad, quad_c, quad_x, quad_y = get_crop_bound(landmark)
            # init crop for detecting face
            bound = np.array([[0, 0], [0, SIZE-1], [SIZE-1, SIZE-1], [SIZE-1, 0]], dtype=np.float32)
            mat = cv2.getAffineTransform(quad[:3], bound[:3])
            img = crop_image(frame, mat, SIZE, SIZE)
            h, w = img.shape[:2]

            boxes = face_boxes(img)
            if len(boxes) == 0:
                continue

            param_lst, roi_box_lst = tddfa(img, boxes)
            box_idx = find_center_bbox(roi_box_lst, w, h)


            param = param_lst[box_idx]
            P = param[:12].reshape(3, -1)  # camera matrix
            s_relative, R, t3d = P2sRt(P)
            pose = matrix2angle(R)
            pose = [p * 180 / np.pi for p in pose]

            # Adjust z-translation in object space
            R_ = param[:12].reshape(3, -1)[:, :3]
            u = tddfa.bfm.u.reshape(3, -1, order='F')
            trans_z = np.array([ 0, 0, 0.5*u[2].mean() ]) # Adjust the object center
            trans = np.matmul(R_, trans_z.reshape(3,1))
            t3d += trans.reshape(3)

            ''' Camera extrinsic estimation for GAN training '''
            # Normalize P to fit in the original image (before 3DDFA cropping)
            sx, sy, ex, ey = roi_box_lst[0]
            scale_x = (ex - sx) / tddfa.size
            scale_y = (ey - sy) / tddfa.size
            t3d[0] = (t3d[0]-1) * scale_x + sx
            t3d[1] = (tddfa.size-t3d[1]) * scale_y + sy
            t3d[0] = (t3d[0] - 0.5*(w-1)) / (0.5*(w-1)) # Normalize to [-1,1]
            t3d[1] = (t3d[1] - 0.5*(h-1)) / (0.5*(h-1)) # Normalize to [-1,1], y is flipped for image space
            t3d[1] *= -1
            t3d[2] = 0 # orthogonal camera is agnostic to Z (the model always outputs 66.67)

            s_relative = s_relative * 2000
            scale_x = (ex - sx) / (w-1)
            scale_y = (ey - sy) / (h-1)
            s = (scale_x + scale_y) / 2 * s_relative
            # print(f"[{iteration}] s={s} t3d={t3d}")

            if s < 0.7 or s > 1.3:
                continue
            if abs(pose[0]) > 90 or abs(pose[1]) > 80 or abs(pose[2]) > 50:
                continue
            if abs(t3d[0]) > 1. or abs(t3d[1]) > 1.:
                continue

            quad_c = quad_c + quad_x * t3d[0]
            quad_c = quad_c - quad_y * t3d[1]
            quad_x = quad_x * s
            quad_y = quad_y * s
            c, x, y = quad_c, quad_x, quad_y
            quad = np.stack([c - x - y, c - x + y, c + x + y, c + x - y]).astype(np.float32)
            if quad[0][0] < -32 or quad[0][1] < -32:
                continue
            orig_size = min(np.linalg.norm(quad[1] - quad[0]), np.linalg.norm(quad[2] - quad[1]))
            if orig_size < SIZE:
                continue

            s = 1
            t3d = 0 * t3d
            R[:,:3] = R[:,:3] * s
            P = np.concatenate([R,t3d[:,None]],1)
            P = np.concatenate([P, np.array([[0,0,0,1.]])], 0)

            pose = eg3dcamparams(P.flatten())
            cropped_img = crop_final(frame, size=SIZE, quad=quad)

            fn = str(frame_idx).zfill(5) + '.jpg'

            labels[fn] = pose
            os.makedirs(image_folder_name, exist_ok=True)
            cv2.imwrite(os.path.join(image_folder_name, fn), cv2.cvtColor(cropped_img, cv2.COLOR_RGB2BGR))

    with open(label_file_name, 'w') as f:
        json.dump(labels, f)

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--in_dir", type=str, required=True)
parser.add_argument("--out_dir", type=str, required=True)

if __name__ == "__main__":
    args = parser.parse_args()
    in_dir = args.in_dir
    assert os.path.exists(in_dir)
    image_out_dir = os.path.join(args.out_dir, 'images')
    label_out_dir = os.path.join(args.out_dir, 'labels')
    os.makedirs(os.path.join(image_out_dir), exist_ok=True)
    os.makedirs(os.path.join(label_out_dir), exist_ok=True)

    video_fn_s = list(filter(lambda f: f.endswith(".mp4"), sorted(os.listdir(in_dir))))

    for fn in tqdm(video_fn_s):
        process_video(os.path.join(in_dir, fn), image_out_dir, label_out_dir)