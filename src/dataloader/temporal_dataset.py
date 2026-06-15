import torch
from torch.utils.data import Dataset
import pandas as pd
import cv2
import numpy as np
import os
import albumentations as A
from albumentations.pytorch import ToTensorV2
import random
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

class TemporalCustomDataset(Dataset):
    """
    시간적 일관성을 고려한 커스텀 데이터셋
    연속된 프레임들을 함께 로드하여 시간적 정보 활용
    """
    
    def __init__(self, df, base_dir, transform=None, sequence_length=3, frame_gap=1):
        """
        Args:
            df: 데이터프레임
            base_dir: 기본 디렉토리
            transform: 변환 함수
            view: 뷰 타입 ('axi' 또는 'sag')
            sequence_length: 시퀀스 길이 (연속된 프레임 수)
            frame_gap: 프레임 간격
        """
        self.df = df
        self.base_dir = base_dir
        self.transform = transform
        self.sequence_length = sequence_length
        self.frame_gap = frame_gap
        
        # 환자별로 프레임 그룹화
        self.patient_frames = self._group_frames_by_patient() # 환자마다 frame들을 모은 dict
        self.valid_starts = self._get_valid_sequence_starts() # frame indexing 저장 [001, 002, 005, ...] -> [0, 1, 2, ...]
    
    def _get_valid_sequence_starts(self):
        """각 환자별로 유효한 시퀀스 시작 인덱스들 반환 (frame_gap=1 전용)"""
        valid_starts = {}
        
        for patient_id, frames in self.patient_frames.items():
            valid_starts[patient_id] = []
            
            if len(frames) >= self.sequence_length:
                # 연속된 sequence_length개 프레임 필요
                max_start_idx = len(frames) - self.sequence_length # 256 - 3 = 253
                for i in range(max_start_idx + 1): # 0 ~ 253 (indexing 저장)
                    valid_starts[patient_id].append(i)
        
        return valid_starts

    def _get_sequence_frames(self, patient_id, start_idx):
        """리스트 인덱스 기준으로 연속 시퀀스 가져오기"""
        frames = self.patient_frames[patient_id]
        sequence_frames = []
        
        for i in range(self.sequence_length):
            idx = start_idx + i
            sequence_frames.append(frames[idx])
        
        return sequence_frames

    def _group_frames_by_patient(self):
        """환자별로 프레임을 그룹화"""
        patient_frames = {}
        
        for _, row in self.df.iterrows():
            # etc_dataset 형식: image_path가 
            # "SUN-Positive/Train/high_grade_adenoma/case_M_20181003094031_0U62363100354631_1_001_002-1_a13_ayy_image0001.jpg" 형태
            patient_id = row['patient_id']  # ex) case_M_20181003094031_0U62363100354631_1_001_002-1_a13_ayy
            frame_num = int(row['frame_num']) # ex) 0001
        
            # 실제 파일 존재 여부 확인 (이미지 + 마스크)
            actual_img_path = os.path.join(self.base_dir, row['image_paths'])
            actual_mask_path = os.path.join(self.base_dir, row['mask_paths'])
            
            if not os.path.exists(actual_img_path):
                raise FileNotFoundError(f"Image file not found: {actual_img_path}")
            if not os.path.exists(actual_mask_path):
                raise FileNotFoundError(f"Mask file not found: {actual_mask_path}")
            
            if patient_id not in patient_frames:
                patient_frames[patient_id] = []
            patient_frames[patient_id].append(frame_num)
        
        # 각 환자의 프레임을 정렬
        for patient_id in patient_frames:
            patient_frames[patient_id].sort()
        
        return patient_frames

    def __len__(self):
        """유효한 시퀀스의 총 개수"""
        return sum(len(starts) for starts in self.valid_starts.values())
    
    def __getitem__(self, idx):
        """데이터 아이템 가져오기"""
        # idx를 환자와 시작 프레임으로 변환
        patient_id, start_frame = self._idx_to_patient_frame(idx) # start frame index
        sequence_frames = self._get_sequence_frames(patient_id, start_frame)
        # print(sequence_frames)
        
        # 연속된 프레임들 로드 (이미지 + 마스크)
        images = []
        labels = []
        
        for frame_num in sequence_frames:
            # 이미지 로드 (etc_dataset 구조)
            row = self.df[(self.df['patient_id']==patient_id) & (self.df['frame_num']==frame_num)].iloc[0]
            img_pre_path = row['image_paths'].split('_image')[0]
            msk_pre_path = row['mask_paths'].split('_image')[0]

            img_path = os.path.join(self.base_dir, img_pre_path+f"_image{frame_num:04d}.jpg")
            
            # 마스크 로드 (etc_dataset 구조)
            mask_path = os.path.join(self.base_dir, msk_pre_path+f"_image{frame_num:04d}.png")
            
            # 파일 존재 여부 확인
            if not os.path.exists(img_path):
                raise FileNotFoundError(f"Image file not found: {img_path}")
            if not os.path.exists(mask_path):
                raise FileNotFoundError(f"Mask file not found: {mask_path}")
            
            # 이미지 로드
            image = cv2.imread(img_path)
            if image is None:
                raise ValueError(f"Failed to load image: {img_path}")
            
            # 마스크 로드 (그레이스케일)
            label = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if label is None:
                raise ValueError(f"Failed to load mask: {mask_path}")
            
            # BGR to RGB 변환
            # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # 마스크 정규화 (0-1 범위)
            label = label.astype(np.float32) / 255.0
            
            # numpy array 타입 확인
            if not isinstance(image, np.ndarray):
                raise TypeError(f"Image is not numpy array: {type(image)}")
            if not isinstance(label, np.ndarray):
                raise TypeError(f"Label is not numpy array: {type(label)}")
            
            if self.transform:
                transformed = self.transform(image=image, mask=label)
                image = transformed['image']
                label = transformed['mask'].unsqueeze(0)
            
            images.append(image)
            labels.append(label)
        
        # 텐서로 변환
        images = torch.stack(images)  # (sequence_length, C, H, W)
        labels = torch.stack(labels)  # (sequence_length, 1, H, W)
        
        return {
            'images': images,  # 연속된 프레임들
            'labels': labels,  # 연속된 마스크들 (pseudo label)
            'patient_id': patient_id,
            'frame_sequence': sequence_frames
        }
    
    def _idx_to_patient_frame(self, idx):
        """인덱스를 환자 ID와 시작 인덱스로 변환"""
        current_idx = 0
        
        for patient_id, starts in self.valid_starts.items(): # patient, frame indexing
            if current_idx + len(starts) > idx:
                local_idx = idx - current_idx
                start_list_idx = starts[local_idx]  # 리스트 상의 시작 인덱스
                return patient_id, start_list_idx
            current_idx += len(starts)
        
        raise IndexError(f"Index {idx} out of range")

def get_temporal_transform(img_size=[256, 448], is_train=True):
    """시간적 데이터셋용 변환 함수"""
    if is_train:
        return A.Compose([
            # A.RandomResizedCrop(img_size[0], img_size[1], scale=(0.9, 1.0)), 
            # A.HorizontalFlip(p=0.5),
            # A.VerticalFlip(p=0.5),
            # A.RandomBrightnessContrast(p=0.5),
            # A.GaussianBlur(p=0.3),
            # A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1,
            #                 rotate_limit=10, p=0.5),
            # A.Cutout(num_holes=4, max_h_size=16, max_w_size=16, p=0.3),
            A.Resize(img_size[0], img_size[1]),
            A.Normalize(),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.Resize(img_size[0], img_size[1]),
            A.Normalize(),
            ToTensorV2(),
        ])

def getDataloader(args):
    img_size = args.img_size
    if args.model == "SwinUnet":
        img_size = [224, 224]
    
    # 시간적 데이터셋용 transform
    train_transform = get_temporal_transform(img_size, is_train=True)
    val_transform = get_temporal_transform(img_size, is_train=False)

    train_df = pd.read_csv(args.csv_dir)
    test_df = pd.read_csv(args.csv_val_dir)

    # 시간적 데이터셋 사용 (view_type에 따라 필터링)
    db_train = TemporalCustomDataset(train_df, args.base_dir, transform=train_transform, 
                                   sequence_length=args.sequence_length, frame_gap=args.frame_gap)
    db_test = TemporalCustomDataset(test_df, args.base_dir, transform=val_transform, 
                                  sequence_length=args.sequence_length, frame_gap=args.frame_gap)

    print("train num:{}, val num:{}, test num:{}".format(len(db_train), len(db_test), len(db_test)))

    trainloader = DataLoader(db_train, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True)
    valloader = DataLoader(db_test, batch_size=args.batch_size, shuffle=False, num_workers=8)
    testloader = DataLoader(db_test, batch_size=args.batch_size, shuffle=False, num_workers=8)

    return trainloader, valloader, testloader

# def generate_flip_pseudo_label(model, images):
#     """
#     Flip-based Pseudo Label 생성
#     원본 이미지와 좌우 반전된 이미지를 모두 모델에 입력하여 더 정확한 pseudo label 생성
#     """
#     batch_size, sequence_length = images.shape[:2]
#     pseudo_labels = []
    
#     for seq_idx in range(sequence_length):
#         img_batch = images[:, seq_idx].cuda()  # (B, C, H, W)
        
#         # 원본 이미지 예측
#         with torch.no_grad():
#             outputs_original = model(img_batch)
#             masks_original = torch.sigmoid(outputs_original)  # (B, 1, H, W)
        
#         # 좌우 반전된 이미지 예측
#         img_flipped = torch.flip(img_batch, dims=[3])  # 좌우 반전
#         with torch.no_grad():
#             outputs_flipped = model(img_flipped)
#             masks_flipped = torch.sigmoid(outputs_flipped)  # (B, 1, H, W)
        
#         # 좌우 반전된 마스크를 다시 원래대로 뒤집기
#         masks_flipped_back = torch.flip(masks_flipped, dims=[3])  # 다시 좌우 반전
        
#         # 원본 마스크와 뒤집힌 마스크의 평균 (더 안정적인 pseudo label)
#         pseudo_mask = (masks_original + masks_flipped_back) / 2.0
        
#         pseudo_labels.append(pseudo_mask)
    
#     # 텐서로 변환
#     pseudo_labels = torch.stack(pseudo_labels, dim=1)  # (B, sequence_length, 1, H, W)
#     return pseudo_labels