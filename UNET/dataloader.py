import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import v2
from pathlib import Path
import numpy as np
from PIL import Image
import cv2
import random


class dEPEDataset(Dataset):
    """Load dEPE dataset images"""

    def __init__(
        self,
        img_dir: list,
        samples: int = 100,
        img_size: int = 200,
        transforms: list = [],
        mask: bool = False,
        crop_params: bool = False,
        hvsem_file="stack_hvsem.tiff",
        top_sem_file="lines_sem.tiff",
        bottom_sem_file="holes_sem.tiff",
        normalize: bool = True,
        blur_and_noise: bool = True,
        elastic_transform: bool = False,
        shift: bool = False
    ):
        self.samples = samples
        self.imgs = img_dir
        self.img_size = img_size
        self.transforms = transforms
        self.mask = mask
        self.crop_params = crop_params
        self.hvsem_file = hvsem_file
        self.top_sem_file = top_sem_file
        self.bottom_sem_file = bottom_sem_file
        self.normalize = normalize
        self.blur_and_noise = blur_and_noise
        self.elastic_transform = elastic_transform
        self.shift = shift

        self.bottoms = []
        self.tops = []


        for idx in range(len(self.imgs)):
            top = v2.functional.pil_to_tensor(
                Image.open(Path.joinpath(self.imgs[idx], self.top_sem_file))
            )
            bottom = v2.functional.pil_to_tensor(
                Image.open(Path.joinpath(self.imgs[idx], self.bottom_sem_file))
            )

            self.tops.append(top)
            self.bottoms.append(bottom)

        if mask:
            h, w = Image.open(Path.joinpath(self.imgs[0], self.hvsem_file)).size

            self.masks = dict()

            for idx in range(len(self.imgs)):

                top_contours = np.load(Path.joinpath(self.imgs[idx], "lines.npz"))
                bottom_contours = np.load(Path.joinpath(self.imgs[idx], "holes.npz"))

                # create top_sem binary image
                top_cont = [top_contours[f] + w / 2 for f in top_contours.files]
                top_sem = np.zeros((w, h), dtype=np.int8)
                for c in top_cont:
                    c = np.vstack((c, c[0]))
                    c = np.array(c).reshape((-1, 1, 2)).astype(np.int32)
                    cv2.drawContours(top_sem, [c], -1, 1, -1) #-1 means thickness --> fills the shape
                    _out_cnt = np.zeros((w, h), dtype=np.int8)
                    cv2.drawContours(_out_cnt, [c], -1, 1, 1)
                    # top_sem -= _out_cnt  # exclude contours line
                top_sem = torch.from_numpy(top_sem)[torch.newaxis, :]

                # create bottom_sem binary image
                bottom_cont = [
                    bottom_contours[f] + w / 2 for f in bottom_contours.files
                ]
                bottom_sem = np.zeros((w, h), dtype=np.int8)
                for c in bottom_cont:
                    c = np.vstack((c, c[0]))
                    c = np.array(c).reshape((-1, 1, 2)).astype(np.int32)
                    cv2.drawContours(bottom_sem, [c], -1, 1, -1)
                    _out_cnt = np.zeros((w, h), dtype=np.int8)
                    cv2.drawContours(_out_cnt, [c], -1, 1, 1)
                    # bottom_sem -= _out_cnt  # exclude contours line
                bottom_sem = torch.from_numpy(bottom_sem)[torch.newaxis, :]

                self.masks.update(
                    {
                        f"{idx}": {
                            "top_cnt": top_cont,
                            "bottom_cnt": bottom_cont,
                            "top_sem": top_sem,
                            "bottom_sem": bottom_sem,
                        }
                    }
                )
            

    def __len__(self):
        return self.samples

    def get_contours(self, idx, i, j, h, w):
        "get contours from idx and crop_params"
        img_dir = self.imgs[int(idx)]
        sem_img_size = Image.open(Path.joinpath(self.imgs[idx], self.hvsem_file)).size[
            0
        ]

        top_contours = np.load(Path.joinpath(img_dir, "lines.npz"))
        top_contours = [top_contours[f] + sem_img_size / 2 for f in top_contours.files]
        s_top_contours = []
        for c in top_contours:
            # cut_idx = int(c.shape[0] / 2)
            # s_top_contours.append(c[:cut_idx, :])
            # s_top_contours.append(c[cut_idx:, :])
            s_top_contours.append(c)

        bottom_contours = np.load(Path.joinpath(img_dir, "holes.npz"))
        bottom_contours = [
            bottom_contours[f] + sem_img_size / 2 for f in bottom_contours.files
        ]

        # Crop Contours
        top_cnt = []
        for c in s_top_contours:
            c = c[(c[:, 0] > j) & (c[:, 0] < j + h) & (c[:, 1] > i) & (c[:, 1] < i + w)]
            c[:, 0] -= j + 0.5
            c[:, 1] -= i + 0.5
            if c.shape[0] > 2:
                top_cnt.append(c)

        bottom_cnt = []
        for c in bottom_contours:
            c = c[(c[:, 0] > j) & (c[:, 0] < j + h) & (c[:, 1] > i) & (c[:, 1] < i + w)]
            c[:, 0] -= j + 0.5
            c[:, 1] -= i + 0.5
            if c.shape[0] > 2:
                bottom_cnt.append(c)

        return top_cnt, bottom_cnt

    def __getitem__(self, idx):
        # Open images as torch tensor
        idx = np.random.randint(0, len(self.imgs))

        top = self.tops[idx]
        bottom = self.bottoms[idx]

        #define top_sem and bottom_sem
        if self.mask:
            _, _, top_sem, bottom_sem = self.masks[str(idx)].values()
        else:
            top_sem = top
            bottom_sem = bottom

        #shift circle
        if self.shift and random.random() < 0.8: #0.8
            shft = random.randint(-15, 15) #random.randint(-15, 15)
            bottom = torch.roll(bottom, shifts= shft, dims=-1)
            bottom_sem = torch.roll(bottom_sem, shifts= shft, dims=-1)
        else:
            shft = 0

        if random.random() < 0.5: #0.5
            hvsem = 0.5 * (bottom + top)
        else:
            hvsem = torch.maximum(bottom, top)
        
        # Image => [hvsem, top_sem, bottom_sem]
        img = torch.vstack([hvsem, top_sem, bottom_sem])
        
        #crop hvsem, top_sem, bottom_sem
        i, j, h, w = v2.RandomCrop.get_params(
            torch.Tensor(hvsem), (self.img_size, self.img_size)
        )
        crop_params = torch.tensor([idx, i, j, h, w], dtype=torch.int16)
        img = v2.functional.crop(img, i, j, h, w)
  
        # apply transforms on images (including [hvsem, top_sem, bottom_sem])
        transforms = v2.Compose(self.transforms + [v2.ToDtype(torch.float32)])
        img = transforms(img)

        # apply elastic transforms on images (including [hvsem, top_sem, bottom_sem])
        if self.elastic_transform and (random.random() < 0.8):
            displacement = elastic_params(h, w, 100., 5.)
        else:
            displacement = torch.zeros(1, h, w, 2)
        img = v2.functional.elastic(
            img,
            displacement=displacement,
        )
        
        hvsem = img[0]
        top_sem, bottom_sem = img[1:]

        # apply blur to hvsem image only
        if self.blur_and_noise:
            noise = v2.RandomApply(
                transforms=[
                    v2.GaussianNoise(mean=0., sigma=0.1, clip=False) #0.25
                ],
                p=0.8,
            )

            blur = v2.RandomApply(
                transforms=[
                    v2.GaussianBlur(kernel_size=5, sigma=(0.1, 1.0)) #(0.5,1.0)
                ],
                p=0.8,
            )

            hvsem = blur(noise(hvsem[None] / 255.))[0]

        # Normalize hvsem or hvsem, top_sem and bottom_sem
        if self.normalize:
            if self.mask:
                std, mean = torch.std_mean(hvsem)
                normalize = v2.Normalize(mean=[mean], std=[std])
                hvsem = normalize(hvsem[None])[0]
            else:
                std, mean = torch.std_mean(img, dim=(1, 2), keepdim=True)
                normalize = v2.Normalize(
                    mean=mean.squeeze().tolist(), std=std.squeeze().tolist()
                )
                img = normalize(img)
                hvsem, top_sem, bottom_sem = img
           
        if self.crop_params:
            return (
                hvsem[torch.newaxis, :],
                top_sem[torch.newaxis, :],
                bottom_sem[torch.newaxis, :],
                crop_params,
                displacement,
                shft
            )
                 
        else:
            return (
                hvsem[torch.newaxis, :],
                top_sem[torch.newaxis, :],
                bottom_sem[torch.newaxis, :],
                displacement,
                shft
            )

def elastic_params(height, width, alpha, sigma) -> torch.Tensor:
    dx = torch.rand(1, 1, height, width) * 2 - 1
    kx = int(8 * sigma + 1)
    # if kernel size is even we have to make it odd
    if kx % 2 == 0:
        kx += 1
    dx = v2.functional.gaussian_blur(dx, [kx, kx], sigma)
    dx = dx * alpha / width

    dy = torch.rand(1, 1, height, width) * 2 - 1
    ky = int(8 * sigma + 1)
    # if kernel size is even we have to make it odd
    if ky % 2 == 0:
        ky += 1
    dy = v2.functional.gaussian_blur(dy, [ky, ky], sigma)
    dy = dy * alpha / height
    displacement = torch.concat([dx, dy], 1).permute([0, 2, 3, 1])  # 1 x H x W x 2
    return displacement

class dEPEDataset_combination(Dataset):
    """Load dEPE dataset images"""

    def __init__(
        self,
        img_dir: list,
        samples: int = 100,
        img_size: int = 200,
        transforms: list = [],
        mask: bool = False,
        crop_params: bool = False,
        hvsem_file="stack_hvsem.tiff",
        top_sem_file="lines_sem.tiff",
        bottom_sem_file="holes_sem.tiff",
        normalize: bool = True  
    ):
        self.samples = samples
        self.imgs = img_dir
        self.img_size = img_size
        self.transforms = transforms
        self.mask = mask
        self.crop_params = crop_params
        self.hvsem_file = hvsem_file
        self.top_sem_file = top_sem_file
        self.bottom_sem_file = bottom_sem_file
        self.normalize = normalize

        if mask:
            h, w = Image.open(Path.joinpath(self.imgs[0], self.hvsem_file)).size

            self.masks = dict()

            for idx in range(len(self.imgs)):

                top_contours = np.load(Path.joinpath(self.imgs[idx], "lines.npz"))
                bottom_contours = np.load(Path.joinpath(self.imgs[idx], "holes.npz"))

                # create top_sem binary image
                top_cont = [top_contours[f] + w / 2 for f in top_contours.files]
                top_sem = np.zeros((w, h), dtype=np.int8)
                for c in top_cont:
                    c = np.vstack((c, c[0]))
                    c = np.array(c).reshape((-1, 1, 2)).astype(np.int32)
                    cv2.drawContours(top_sem, [c], -1, 1, -1) #-1 means thickness --> fills the shape
                    _out_cnt = np.zeros((w, h), dtype=np.int8)
                    cv2.drawContours(_out_cnt, [c], -1, 1, 1)
                    # top_sem -= _out_cnt  # exclude contours line
                top_sem = torch.from_numpy(top_sem)[torch.newaxis, :]

                # create bottom_sem binary image
                bottom_cont = [
                    bottom_contours[f] + w / 2 for f in bottom_contours.files
                ]
                bottom_sem = np.zeros((w, h), dtype=np.int8)
                for c in bottom_cont:
                    c = np.vstack((c, c[0]))
                    c = np.array(c).reshape((-1, 1, 2)).astype(np.int32)
                    cv2.drawContours(bottom_sem, [c], -1, 1, -1)
                    _out_cnt = np.zeros((w, h), dtype=np.int8)
                    cv2.drawContours(_out_cnt, [c], -1, 1, 1)
                    # bottom_sem -= _out_cnt  # exclude contours line
                bottom_sem = torch.from_numpy(bottom_sem)[torch.newaxis, :]

                self.masks.update(
                    {
                        f"{idx}": {
                            "top_cnt": top_cont,
                            "bottom_cnt": bottom_cont,
                            "top_sem": top_sem,
                            "bottom_sem": bottom_sem,
                        }
                    }
                )

    def __len__(self):
        return self.samples

    def get_contours(self, idx, i, j, h, w):
        "get contours from idx and crop_params"
        img_dir = self.imgs[int(idx)]
        sem_img_size = Image.open(Path.joinpath(self.imgs[idx], self.hvsem_file)).size[
            0
        ]

        top_contours = np.load(Path.joinpath(img_dir, "lines.npz"))
        top_contours = [top_contours[f] + sem_img_size / 2 for f in top_contours.files]
        s_top_contours = []
        for c in top_contours:
            cut_idx = int(c.shape[0] / 2)
            s_top_contours.append(c[:cut_idx, :])
            s_top_contours.append(c[cut_idx:, :])

        bottom_contours = np.load(Path.joinpath(img_dir, "holes.npz"))
        bottom_contours = [
            bottom_contours[f] + sem_img_size / 2 for f in bottom_contours.files
        ]

        # Crop Contours
        top_cnt = []
        for c in s_top_contours:
            c = c[(c[:, 0] > j) & (c[:, 0] < j + h) & (c[:, 1] > i) & (c[:, 1] < i + w)]
            c[:, 0] -= j + 0.5
            c[:, 1] -= i + 0.5
            if c.shape[0] > 2:
                top_cnt.append(c)

        bottom_cnt = []
        for c in bottom_contours:
            c = c[(c[:, 0] > j) & (c[:, 0] < j + h) & (c[:, 1] > i) & (c[:, 1] < i + w)]
            c[:, 0] -= j + 0.5
            c[:, 1] -= i + 0.5
            if c.shape[0] > 2:
                bottom_cnt.append(c)

        return top_cnt, bottom_cnt

    def __getitem__(self, idx):
        # Open images as torch tensor
        idx = np.random.randint(0, len(self.imgs))

        # Image => [hvsem, top_sem, bottom_sem]
        hvsem = v2.functional.pil_to_tensor(
            Image.open(Path.joinpath(self.imgs[idx], self.hvsem_file))
        )

        hvsem_top = v2.functional.pil_to_tensor(
            Image.open(Path.joinpath(self.imgs[idx], self.top_sem_file))
        )

        hvsem_bottom = v2.functional.pil_to_tensor(
            Image.open(Path.joinpath(self.imgs[idx], self.bottom_sem_file))
        )

        i, j, h, w = v2.RandomCrop.get_params(
            torch.Tensor(hvsem), (self.img_size, self.img_size)
        )

        crop_params = torch.tensor([idx, i, j, h, w], dtype=torch.int16)

        hvsem = v2.functional.crop(hvsem, i, j, h, w)
        hvsem_top = v2.functional.crop(hvsem_top, i, j, h, w)
        hvsem_bottom = v2.functional.crop(hvsem_bottom, i, j, h, w)

        if self.mask:
            _, _, top_sem, bottom_sem = self.masks[str(idx)].values()

            # background = 1 - torch.max(torch.cat((top_sem, bottom_sem), dim=0), dim=0, keepdim=True).values
            
            top_sem = v2.functional.crop(top_sem, i, j, h, w)
            bottom_sem = v2.functional.crop(bottom_sem, i, j, h, w)
            # background = v2.functional.crop(background, i, j, h, w)

            img = torch.vstack([hvsem, hvsem_top, hvsem_bottom, top_sem, bottom_sem])
        else:
            top_sem = v2.functional.pil_to_tensor(
                Image.open(Path.joinpath(self.imgs[idx], self.top_sem_file))
            )
            bottom_sem = v2.functional.pil_to_tensor(
                Image.open(Path.joinpath(self.imgs[idx], self.bottom_sem_file))
            )

            top_sem = v2.functional.crop(top_sem, i, j, h, w)
            bottom_sem = v2.functional.crop(bottom_sem, i, j, h, w)

            img = torch.vstack([hvsem, hvsem_top, hvsem_bottom, top_sem, bottom_sem])

    
        # apply transforms
        transforms = v2.Compose(self.transforms + [v2.ToDtype(torch.float32)])

        img = transforms(img)

        # apply blur to hvsem image only
        blur = v2.RandomApply(
            transforms=[
                v2.GaussianBlur(kernel_size=k, sigma=(0.5, 1.0)) for k in [3, 5]
            ],
            p=0.8,
        )
        
        img = torch.vstack([blur(img[0][torch.newaxis, :]), blur(img[1][torch.newaxis, :]), blur(img[2][torch.newaxis, :]), img[3:]])
        
        hvsem = img[0][torch.newaxis, :][0]
        hvsem_top = img[1][torch.newaxis, :][0]
        hvsem_bottom = img[2][torch.newaxis, :][0]
        top_sem, bottom_sem = img[3:]

        # Normalize
        if self.normalize:
            if self.mask:
                std, mean = torch.std_mean(img[:3], dim=(1, 2), keepdim=True)
                normalize = v2.Normalize(
                    mean=mean.squeeze().tolist(), std=std.squeeze().tolist()
                )
                img_normalized = normalize(img[:3])

                hvsem, hvsem_top, hvsem_bottom = img_normalized[:3]
                
                top_sem, bottom_sem = img[3:]
            else:
                std, mean = torch.std_mean(img, dim=(1, 2), keepdim=True)
                normalize = v2.Normalize(
                    mean=mean.squeeze().tolist(), std=std.squeeze().tolist()
                )
                img = normalize(img)
                hvsem, hvsem_top, hvsem_bottom, top_sem, bottom_sem = img

        alpha_1 = np.random.rand()
        beta_1 = 1 - alpha_1

        alpha_2 = np.random.rand()
        beta_2 = 1 - alpha_2
    
        if self.crop_params:
            return (
                # (1-(1-hvsem_top) * (1-hvsem_bottom))[torch.newaxis, :],
                (alpha_2 * (alpha_1 * hvsem_top + beta_1 * hvsem_bottom) + beta_2 * hvsem)[torch.newaxis, :],
                #(torch.max(torch.stack(((alpha * hvsem_top), (beta * hvsem_bottom)),dim=0), dim=0).values)[torch.newaxis, :],
                top_sem[torch.newaxis, :],
                bottom_sem[torch.newaxis, :],
                crop_params,
            )
                
        else:
            return (
                (alpha_2 * (alpha_1 * hvsem_top + beta_1 * hvsem_bottom) + beta_2 * hvsem)[torch.newaxis, :],
                #(torch.max(torch.stack(((alpha * hvsem_top), (beta * hvsem_bottom)),dim=0), dim=0).values)[torch.newaxis, :],
                top_sem[torch.newaxis, :],
                bottom_sem[torch.newaxis, :],
            
            )


class dEPEDataset_backup(Dataset):
    """Load dEPE dataset images"""

    def __init__(
        self,
        img_dir,
        samples=100,
        img_size=200,
        tfsm=[],
        normalize=True,
        blur=True,
        output_mask=False,
        crop_params=False,
        crop_border=False,
        hvsem_file="stack_hvsem.tiff",
        top_sem_file="lines_sem.tiff",
        bottom_sem_file="holes_sem.tiff",
    ):
        self.samples = samples
        if isinstance(img_dir, list):
            self.imgs = img_dir
        else:
            self.imgs = [d for d in Path(img_dir).iterdir() if d.is_dir()]
        self.img_size = img_size
        self.tfsm = tfsm
        self.normalize = normalize
        self.blur = blur
        self.output_mask = output_mask
        self.crop_params = crop_params
        self.crop_border = crop_border
        self.hvsem_file = hvsem_file
        self.top_sem_file = top_sem_file
        self.bottom_sem_file = bottom_sem_file

    def __len__(self):
        return self.samples

    def get_contours(self, idx, i, j, h, w):
        "get contours from idx and crop_params"
        img_dir = self.imgs[int(idx)]
        sem_img_size = Image.open(Path.joinpath(self.imgs[idx], self.hvsem_file)).size[
            0
        ]

        top_contours = np.load(Path.joinpath(img_dir, "lines.npz"))
        top_contours = [top_contours[f] + sem_img_size / 2 for f in top_contours.files]
        s_top_contours = []
        for c in top_contours:
            cut_idx = int(c.shape[0] / 2)
            s_top_contours.append(c[:cut_idx, :])
            s_top_contours.append(c[cut_idx:, :])

        bottom_contours = np.load(Path.joinpath(img_dir, "holes.npz"))
        bottom_contours = [
            bottom_contours[f] + sem_img_size / 2 for f in bottom_contours.files
        ]

        # Crop Contours
        top_cnt = []
        for c in s_top_contours:
            c = c[(c[:, 0] > j) & (c[:, 0] < j + h) & (c[:, 1] > i) & (c[:, 1] < i + w)]
            c[:, 0] -= j + 0.5
            c[:, 1] -= i + 0.5
            if c.shape[0] > 2:
                top_cnt.append(c)

        bottom_cnt = []
        for c in bottom_contours:
            c = c[(c[:, 0] > j) & (c[:, 0] < j + h) & (c[:, 1] > i) & (c[:, 1] < i + w)]
            c[:, 0] -= j + 0.5
            c[:, 1] -= i + 0.5
            if c.shape[0] > 2:
                bottom_cnt.append(c)

        return top_cnt, bottom_cnt

    def __getitem__(self, idx):
        # Open images as torch tensor
        idx = np.random.randint(0, len(self.imgs))

        # Open contours
        top_contours = np.load(Path.joinpath(self.imgs[idx], "lines.npz"))
        bottom_contours = np.load(Path.joinpath(self.imgs[idx], "holes.npz"))

        # Open images
        stack_img = np.array(Image.open(Path.joinpath(self.imgs[idx], self.hvsem_file)))
        width, height = stack_img.shape

        # Random Crop Image
        i, j, h, w = transforms.RandomCrop.get_params(
            torch.Tensor(stack_img), (self.img_size, self.img_size)
        )

        # if self.crop_border:
        #     b = 50
        #     i = np.random.randint(b, (width - (self.img_size + b)))
        #     j = np.random.randint(b, (width - (self.img_size + b)))
        # else:
        #     i = np.random.randint(0, width - self.img_size)
        #     j = np.random.randint(0, width - self.img_size)

        h = w = self.img_size

        crop_params = torch.Tensor([idx, i, j, h, w]).to(torch.int16)

        stack_img = transforms.functional.crop(torch.Tensor(stack_img), i, j, h, w)

        if self.output_mask:
            # Create top_img mask mask
            top_cont = [top_contours[f] + width / 2 for f in top_contours.files]
            top_mask = np.zeros((width, height), dtype=np.int8)
            for c in top_cont:
                c = np.vstack((c, c[0]))
                c = np.array(c).reshape((-1, 1, 2)).astype(np.int32)
                cv2.drawContours(top_mask, [c], -1, 1, -1)
                _out_cnt = np.zeros((width, height), dtype=np.int8)
                cv2.drawContours(_out_cnt, [c], -1, 1, 1)
                top_mask -= _out_cnt

            top_img = transforms.functional.crop(torch.Tensor(top_mask), i, j, h, w)

            # Create bottom_img mask mask
            bottom_cont = [
                bottom_contours[f] + width / 2 for f in bottom_contours.files
            ]
            bottom_mask = np.zeros((width, height), dtype=np.int8)
            for c in bottom_cont:
                c = np.vstack((c, c[0]))
                c = np.array(c).reshape((-1, 1, 2)).astype(np.int32)
                cv2.drawContours(bottom_mask, [c], -1, 1, -1)
                _out_cnt = np.zeros((width, height), dtype=np.int8)
                cv2.drawContours(_out_cnt, [c], -1, 1, 1)
                bottom_mask -= _out_cnt

            bottom_img = transforms.functional.crop(
                torch.Tensor(bottom_mask), i, j, h, w
            )

        else:
            top_img = np.array(
                Image.open(Path.joinpath(self.imgs[idx], self.top_sem_file))
            )
            bottom_img = np.array(
                Image.open(Path.joinpath(self.imgs[idx], self.bottom_sem_file))
            )

            top_img = transforms.functional.crop(torch.Tensor(top_img), i, j, h, w)

            bottom_img = transforms.functional.crop(
                torch.Tensor(bottom_img), i, j, h, w
            )

        # Gaussian Blur only on hvsem input image
        if self.blur:
            blur = transforms.RandomApply(
                transforms=[
                    transforms.GaussianBlur(kernel_size=k, sigma=(0.1, 5.0))
                    for k in [1, 3]
                ],
                p=0.66,
            )
            stack_img = blur(stack_img[torch.newaxis, :])[0]

        image = torch.Tensor(
            np.stack([stack_img, top_img, bottom_img])[np.newaxis, :, :, :]
        )

        # Normalize
        if self.normalize:
            if self.output_mask:
                norm = transforms.Normalize(
                    mean=(torch.mean(image[:, 0])),
                    std=(torch.std(image[:, 0])),
                )
                image[:, 0] = norm(image[:, 0])
            else:
                norm = transforms.Normalize(
                    mean=torch.mean(image, axis=(2, 3)).numpy()[0],
                    std=torch.std(image, axis=(2, 3)).numpy()[0],
                )
                image = norm(image)

        # Apply transforms
        tfsm = transforms.Compose(self.tfsm)
        image = tfsm(image)

        if self.crop_params:
            return (
                image[:, 0, :, :],  # hvsem
                image[:, 1, :, :],  # top_sem
                image[:, 2, :, :],  # bottom_sem
                crop_params,
            )
        else:
            return (
                image[:, 0, :, :],  # hvsem
                image[:, 1, :, :],  # top_sem
                image[:, 2, :, :],  # bottom_sem
            )
