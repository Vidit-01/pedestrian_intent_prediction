"""Dataset for Pedestrian Graph+ over the PCPA sample pickle + the cache written by extract.py.

Returns per window:  kp [3,T,19] (x/W, y/H, score), frame [3,192,64] (local-context crop of the last observed frame, ImageNet-normalised),
                     vel [2,T] (JAAD vehicle-action category 0..4 tiled to 2 channels, as the released code does), y.
Training augmentation follows the released recipe: keypoint jitter, the heavy photometric torchvision stack on the crop, and
(the optional 'time crop' is applied per batch in train.py).
"""
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms as A

MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


def photometric_train():
    return A.Compose([A.ToPILImage(), A.RandomPosterize(bits=2), A.RandomInvert(p=0.2), A.RandomSolarize(threshold=50.0),
                      A.RandomAdjustSharpness(sharpness_factor=2), A.RandomAutocontrast(p=0.2), A.RandomEqualize(p=0.2),
                      A.ColorJitter(0.5, 0.3), A.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2)), A.ToTensor(),
                      A.Normalize(MEAN, STD)])


def photometric_eval():
    return A.Compose([A.ToPILImage(), A.ToTensor(), A.Normalize(MEAN, STD)])


class PedGraphDataset(Dataset):
    def __init__(self, rec, cache_dir, train=False, jitter=(4, 3), use_image=True, use_vel=True):
        self.rec, self.dir, self.train = rec, cache_dir, train
        self.jitter, self.use_image, self.use_vel = jitter, use_image, use_vel
        self.tf = photometric_train() if train else photometric_eval()
        self._c = {}

    def __len__(self):
        return len(self.rec['label'])

    def _video(self, vid):
        if vid not in self._c:
            z = np.load(f'{self.dir}/{vid}.npz')
            pids = list(z['pids'])
            self._c[vid] = ({(pids[p], int(f)): r for r, (p, f) in enumerate(z['keys'])},
                            {(pids[p], int(f)): r for r, (p, f) in enumerate(z['crop_keys'])}, z['kps'], z['crops'])
        return self._c[vid]

    def __getitem__(self, i):
        rec = self.rec
        vid, pid = rec['vid'][i], str(rec['pid'][i])
        kix, cix, kps, crops = self._video(vid)
        frames = [int(f) for f in rec['frames'][i]]
        kp = np.stack([kps[kix[(pid, f)]] for f in frames]).copy()          # [T,19,3]
        W, H = rec['img_wh'][i]
        if self.train and self.jitter:
            kp[..., 0] += np.random.randint(-self.jitter[0], self.jitter[0] + 1, kp[..., 0].shape)
            kp[..., 1] += np.random.randint(-self.jitter[1], self.jitter[1] + 1, kp[..., 1].shape)
        kp[..., 0] = np.clip(kp[..., 0], 0, W) / W
        kp[..., 1] = np.clip(kp[..., 1], 0, H) / H
        kp = torch.from_numpy(kp.transpose(2, 0, 1)).float().contiguous()   # [3,T,19]
        out = {'kp': kp, 'y': torch.tensor(float(rec['label'][i])), 'idx': i}
        if self.use_image:
            out['frame'] = self.tf(crops[cix[(pid, frames[-1])]]).contiguous()
        if self.use_vel:
            v = np.tile(rec['vehicle_act'][i].astype(np.float32), [2, 1])    # [2,T]
            out['vel'] = torch.from_numpy(v)
        return out
