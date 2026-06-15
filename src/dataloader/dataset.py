import os
import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
from glob import glob
from PIL import Image
import pydicom

class MedicalDataSets(Dataset):
    def __init__(
            self,
            base_dir=None,
            split="train",
            transform=None,
            train_file_dir="train.txt",
            val_file_dir="val.txt",
            n_classes=1,
    ):
        self._base_dir = base_dir
        self.sample_list = []
        self.split = split
        self.transform = transform
        self.train_list = []
        self.semi_list = []
        self.n_classes = n_classes

        if self.split == "train":
            with open(os.path.join(self._base_dir, train_file_dir), "r") as f1:
                self.sample_list = f1.readlines()
            self.sample_list = [item.replace("\n", "") for item in self.sample_list]

        elif self.split == "val":
            with open(os.path.join(self._base_dir, val_file_dir), "r") as f:
                self.sample_list = f.readlines()
            self.sample_list = [item.replace("\n", "") for item in self.sample_list]

        print("total {}  {} samples".format(len(self.sample_list), self.split))

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):

        case = self.sample_list[idx]

        image = cv2.imread(os.path.join(self._base_dir, 'images', case + '.png'))
        if self.n_classes == 1:
            label = \
                cv2.imread(os.path.join(self._base_dir, 'masks', '0', case + '.png'), cv2.IMREAD_GRAYSCALE)[
                    ..., None]
        else:
            label = cv2.imread(os.path.join(self._base_dir, 'masks', '0', case + '.png'), cv2.IMREAD_GRAYSCALE)[..., None]
            for class_idx in range(1, self.n_classes):
                label = np.concatenate((label, cv2.imread(os.path.join(self._base_dir, 'masks', str(class_idx), case + '.png'), cv2.IMREAD_GRAYSCALE)[..., None]), axis=-1)

        augmented = self.transform(image=image, mask=label)
        image = augmented['image']
        label = augmented['mask']

        image = image.astype('float32') / 255
        image = image.transpose(2, 0, 1)

        label = label.astype('float32') / 255
        label = label.transpose(2, 0, 1)

        sample = {"image": image, "label": label, "idx": idx}
        return sample

class CustomDataset(Dataset):
  def __init__(self, df, root_dir, transform=None):
      self.patient_df = df
      self.root_dir = root_dir
      self.transform = transform
      
  def __len__(self):
      return len(self.patient_df)

  def __getitem__(self, idx):
    if torch.is_tensor(idx):
        idx = idx.tolist()

    current_patient = self.patient_df.iloc[idx]
    axi_img_path = current_patient['axi_img']
    axi_msk_path = current_patient['axi_msk']
    sag_img_path = current_patient['sag_img']
    sag_msk_path = current_patient['sag_msk']

    axi = cv2.imread(os.path.join(self.root_dir, axi_img_path))
    axi_msk = cv2.imread(os.path.join(self.root_dir, axi_msk_path), cv2.IMREAD_GRAYSCALE)
    sag = cv2.imread(os.path.join(self.root_dir, sag_img_path))
    sag_msk = cv2.imread(os.path.join(self.root_dir, sag_msk_path), cv2.IMREAD_GRAYSCALE)

    axi_augmented = self.transform(image=axi, mask=axi_msk)

    axi = axi_augmented['image']
    axi_msk = axi_augmented['mask']

    # axi = axi.astype('float32') / 255
    axi = axi.transpose(2, 0, 1)

    axi_msk = axi_msk.astype('float32') / 255

    sag_augmented = self.transform(image=sag, mask=sag_msk)

    sag = sag_augmented['image']
    sag_msk = sag_augmented['mask']

    # sag = sag.astype('float32') / 255
    sag = sag.transpose(2, 0, 1)

    sag_msk = sag_msk.astype('float32') / 255

    label = np.stack((axi_msk, sag_msk)) # (2, 256, 256)

    sample = {
        'AXI': axi, 
        'SAG': sag,
        'LABEL': label,
        "idx": idx
    }
    return sample
  

class CustomDataset_single(Dataset):
  def __init__(self, df, root_dir, transform=None, view='axi'):
      self.patient_df = df
      self.root_dir = root_dir
      self.transform = transform
      self.view = view
      
  def __len__(self):
      return len(self.patient_df)

  def __getitem__(self, idx):
    if torch.is_tensor(idx):
        idx = idx.tolist()

    current_patient = self.patient_df.iloc[idx]
    axi_img_path = current_patient['axi_img']
    axi_msk_path = current_patient['axi_msk']
    sag_img_path = current_patient['sag_img']
    sag_msk_path = current_patient['sag_msk']

    axi = cv2.imread(os.path.join(self.root_dir, axi_img_path))
    axi_msk = cv2.imread(os.path.join(self.root_dir, axi_msk_path), cv2.IMREAD_GRAYSCALE)
    sag = cv2.imread(os.path.join(self.root_dir, sag_img_path))
    sag_msk = cv2.imread(os.path.join(self.root_dir, sag_msk_path), cv2.IMREAD_GRAYSCALE)

    axi_augmented = self.transform(image=axi, mask=axi_msk)

    axi = axi_augmented['image']
    axi_msk = axi_augmented['mask']

    # axi = axi.astype('float32') / 255
    axi = axi.transpose(2, 0, 1)

    axi_msk = axi_msk.astype('float32') / 255

    sag_augmented = self.transform(image=sag, mask=sag_msk)

    sag = sag_augmented['image']
    sag_msk = sag_augmented['mask']

    # sag = sag.astype('float32') / 255
    sag = sag.transpose(2, 0, 1)

    sag_msk = sag_msk.astype('float32') / 255

    # label = np.stack((axi_msk, sag_msk)) # (2, 256, 256)

    if self.view == 'axi':
        sample = {
            'image': axi, 
            'label': np.expand_dims(axi_msk, 0),
        }
    else:
        sample = {
            'image': sag, 
            'label': np.expand_dims(sag_msk, 0),
        }
    return sample
  
class ExtraValidationDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        self.img_paths = glob(self.root_dir + '/*.png')
        
    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):

        img_path = self.img_paths[idx]
        image = cv2.imread(img_path)
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]  # (H, W, C), float32, normalized
            # image = image.astype('float32') / 255
            image = image.transpose(2, 0, 1)
        return image, img_path

class TestDCMDataset(Dataset):
    def __init__(self, root_dir, transform=None, view='axi'):
        self.root_dir = root_dir
        self.transform = transform
        if view=='axi':
            self.dcm_paths = glob(self.root_dir + '/*a.dcm')
        else:
            self.dcm_paths = glob(self.root_dir + '/*s.dcm')
        
    def __len__(self):
        return len(self.dcm_paths)

    def __getitem__(self, idx):

        dcm_path = self.dcm_paths[idx]
        ds = pydicom.filereader.dcmread(dcm_path)
        original_img = ds.pixel_array
        if hasattr(ds, "SequenceOfUltrasoundRegions"):
            region = ds.SequenceOfUltrasoundRegions[0]
            region_x0 = getattr(region, "RegionLocationMinX0", 0)
            region_y0 = getattr(region, "RegionLocationMinY0", 0)
            region_x1 = getattr(region, "RegionLocationMaxX1", 0)
            region_y1 = getattr(region, "RegionLocationMaxY1", 0)
            physical_delta_x = getattr(region, "PhysicalDeltaX", 0) * 10
            physical_delta_y = getattr(region, "PhysicalDeltaY", 0) * 10
            original_img = original_img[region_y0:region_y1, region_x0:region_x1]
            img_normalized = cv2.normalize(original_img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            origin_h, origin_w = original_img.shape[:2]
        else:
            print("No SequenceOfUltrasoundRegions found. Using entire image.")

        patient_id = dcm_path.split('/')[-1].split('_')[0]
        print(img_normalized.shape)
        if self.transform:
            augmented = self.transform(image=img_normalized)
            image = augmented["image"]  # (H, W, C), float32, normalized
            # image = image.astype('float32') / 255
            image = image.transpose(2, 0, 1)
        return image, img_normalized, patient_id, physical_delta_x, physical_delta_y, origin_h, origin_w