import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import pandas as pd
import cv2
import numpy as np
import os
import albumentations as A
from albumentations.pytorch import ToTensorV2

class TemporalCustomDataset(Dataset):
    """
    시간적 일관성을 고려한 커스텀 데이터셋
    연속된 프레임들을 함께 로드하여 시간적 정보 활용
    """
    
    def __init__(self, df, base_dir, transform=None, sequence_length=3, frame_gap=1, dataset_type='SUN', return_path=False):
        """
        Args:
            df: 데이터프레임
            base_dir: 기본 디렉토리
            transform: 변환 함수
            sequence_length: 시퀀스 길이 (연속된 프레임 수)
            frame_gap: 프레임 간격
        """
        self.df = df
        self.base_dir = base_dir
        self.transform = transform
        self.sequence_length = sequence_length
        self.frame_gap = frame_gap
        self.dataset_type = dataset_type
        
        # 환자별로 프레임 그룹화
        self.patient_frames = self._group_frames_by_patient()
        self.valid_starts = self._get_valid_sequence_starts()

        self.return_path = return_path
    
    def _get_valid_sequence_starts(self):
        """각 환자별로 유효한 시퀀스 시작 인덱스들 반환"""
        valid_starts = {}
        
        for patient_id, frames in self.patient_frames.items():
            valid_starts[patient_id] = []
            
            if len(frames) >= self.sequence_length:
                max_start_idx = len(frames) - self.sequence_length
                for i in range(max_start_idx + 1):
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
            patient_id = row['patient_id']
            frame_num = int(row['frame_num'])
        
            # 실제 파일 존재 여부 확인
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
        patient_id, start_frame = self._idx_to_patient_frame(idx)
        sequence_frames = self._get_sequence_frames(patient_id, start_frame)
        
        # 연속된 프레임들 로드
        images = []
        labels = []
        image_paths= []
        
        for frame_num in sequence_frames:
            row = self.df[(self.df['patient_id']==patient_id) & (self.df['frame_num']==frame_num)].iloc[0]

            if self.dataset_type == 'SUN':
                img_pre_path = row['image_paths'].split('_image')[0]
                msk_pre_path = row['mask_paths'].split('_image')[0]

                img_path = os.path.join(self.base_dir, img_pre_path+f"_image{frame_num:04d}.jpg")
                mask_path = os.path.join(self.base_dir, msk_pre_path+f"_image{frame_num:04d}.png")
            
            # /home/work/DY_JH/DY/Datasets/TRUS_V/images/13810746_0_axi.png,
            elif self.dataset_type == 'TRUS':
                img_pre_path = row['image_paths'].split('/')[-1].split('_')[0] # id
                msk_pre_path = row['mask_paths'].split('/')[-1].split('_')[0]

                img_path = os.path.join(self.base_dir, "images", img_pre_path+f"_{frame_num}_{patient_id.split('_')[-1]}.png")
                mask_path = os.path.join(self.base_dir, "masks", msk_pre_path+f"_{frame_num}_{patient_id.split('_')[-1]}.png")

            if not os.path.exists(img_path):
                raise FileNotFoundError(f"Image file not found: {img_path}")
            if not os.path.exists(mask_path):
                raise FileNotFoundError(f"Mask file not found: {mask_path}")
            
            # 이미지 로드
            image = cv2.imread(img_path)
            if image is None:
                raise ValueError(f"Failed to load image: {img_path}")
            
            # 마스크 로드
            label = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if label is None:
                raise ValueError(f"Failed to load mask: {mask_path}")
            
            # 마스크 정규화
            label = label.astype(np.float32) / 255.0
            
            if self.transform:
                transformed = self.transform(image=image, mask=label)
                image = transformed['image']
                label = transformed['mask'].unsqueeze(0)
            
            images.append(image)
            labels.append(label)
            image_paths.append(row['image_paths'])
        
        # 텐서로 변환
        images = torch.stack(images)  # (T, C, H, W)
        labels = torch.stack(labels)  # (T, 1, H, W)
        
        if self.return_path:
            return {
                'images': images,
                'labels': labels,
                'patient_id': patient_id,
                'frame_sequence': sequence_frames,
                'image_paths': image_paths
            }
        else:
            return {
                'images': images,
                'labels': labels,
                'patient_id': patient_id,
                'frame_sequence': sequence_frames
            }
    
    def _idx_to_patient_frame(self, idx):
        """인덱스를 환자 ID와 시작 인덱스로 변환"""
        current_idx = 0
        
        for patient_id, starts in self.valid_starts.items():
            if current_idx + len(starts) > idx:
                local_idx = idx - current_idx
                start_list_idx = starts[local_idx]
                return patient_id, start_list_idx
            current_idx += len(starts)
        
        raise IndexError(f"Index {idx} out of range")


def get_temporal_transform(img_size=[256, 448], is_train=True):
    """시간적 데이터셋용 변환 함수"""
    if is_train:
        return A.Compose([
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
    """
    DDP를 지원하는 DataLoader 생성 함수
    
    Args:
        args: 설정 인자
            - multi_gpu (bool): Multi-GPU 모드 여부
            - rank (int): 현재 프로세스의 rank
            - world_size (int): 전체 프로세스 수
    """
    img_size = args.img_size
    if args.model == "SwinUnet":
        img_size = [224, 224]
    
    train_transform = get_temporal_transform(img_size, is_train=True)
    val_transform = get_temporal_transform(img_size, is_train=False)

    train_df = pd.read_csv(args.csv_dir)
    test_df = pd.read_csv(args.csv_val_dir) if args.csv_val_dir else train_df

    # 데이터셋 생성
    db_train = TemporalCustomDataset(
        train_df, args.base_dir, transform=train_transform,
        sequence_length=args.sequence_length, frame_gap=args.frame_gap, dataset_type=args.dataset
    )
    db_test = TemporalCustomDataset(
        test_df, args.base_dir, transform=val_transform,
        sequence_length=args.sequence_length, frame_gap=args.frame_gap, dataset_type=args.dataset, return_path=True
    )

    print(f"Dataset sizes - Train: {len(db_train)}, Val: {len(db_test)}")

    # DDP 모드 확인: multi_gpu 속성과 world_size > 1 체크
    use_ddp = (hasattr(args, 'multi_gpu') and args.multi_gpu and 
               hasattr(args, 'world_size') and args.world_size > 1)
    
    if use_ddp:
        train_sampler = DistributedSampler(
            db_train,
            num_replicas=args.world_size,
            rank=args.rank,
            shuffle=True,
            seed=args.seed
        )
        val_sampler = DistributedSampler(
            db_test,
            num_replicas=args.world_size,
            rank=args.rank,
            shuffle=False
        )
        
        trainloader = DataLoader(
            db_train,
            batch_size=args.batch_size,
            sampler=train_sampler,
            num_workers=8,
            pin_memory=True,
            drop_last=True  # DDP에서 배치 크기 불일치 방지
        )
        valloader = DataLoader(
            db_test,
            batch_size=args.batch_size,
            sampler=val_sampler,
            num_workers=8,
            pin_memory=True
        )
        testloader = valloader
    else:
        # Single GPU 모드
        trainloader = DataLoader(
            db_train,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=8,
            pin_memory=True
        )
        valloader = DataLoader(
            db_test,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=8,
            pin_memory=True
        )
        testloader = valloader

    return trainloader, valloader, testloader