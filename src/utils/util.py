import argparse
import torch
import cv2
import numpy as np
from scipy import ndimage

def str2bool(v):
    if v.lower() in ['true', 1]:
        return True
    elif v.lower() in ['false', 0]:
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# 간단한 Optical Flow 구현 (FlowNet2 대체)
import torch.nn as nn
import torch.nn.functional as F

class SimpleOpticalFlow(nn.Module):
    """개선된 Optical Flow 계산"""
    def __init__(self):
        super(SimpleOpticalFlow, self).__init__()
        
    def forward(self, img1, img2):
        """
        개선된 optical flow 계산
        img1, img2: (B, C, H, W) 형태의 이미지
        return: (B, 2, H, W) 형태의 flow
        """
        B, C, H, W = img1.size()
        
        # 이미지를 그레이스케일로 변환 (flow 계산용)
        if C == 3:
            gray1 = 0.299 * img1[:, 0] + 0.587 * img1[:, 1] + 0.114 * img1[:, 2]
            gray2 = 0.299 * img2[:, 0] + 0.587 * img2[:, 1] + 0.114 * img2[:, 2]
        else:
            gray1 = img1.squeeze(1)
            gray2 = img2.squeeze(1)
        
        # Sobel 필터로 gradient 계산
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3).to(img1.device)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3).to(img1.device)
        
        # Gradient 계산
        Ix = F.conv2d(gray1.unsqueeze(1), sobel_x, padding=1)
        Iy = F.conv2d(gray1.unsqueeze(1), sobel_y, padding=1)
        It = gray2.unsqueeze(1) - gray1.unsqueeze(1)
        
        # Lucas-Kanade 방법 기반 flow 계산
        # A^T * A * [u, v] = A^T * b
        A11 = (Ix * Ix).sum(dim=(2, 3), keepdim=True)
        A12 = (Ix * Iy).sum(dim=(2, 3), keepdim=True)
        A21 = A12
        A22 = (Iy * Iy).sum(dim=(2, 3), keepdim=True)
        
        b1 = -(Ix * It).sum(dim=(2, 3), keepdim=True)
        b2 = -(Iy * It).sum(dim=(2, 3), keepdim=True)
        
        # 행렬 역행렬 계산
        det = A11 * A22 - A12 * A21
        det = torch.clamp(det, min=1e-6)  # 수치 안정성
        
        u = (A22 * b1 - A12 * b2) / det
        v = (-A21 * b1 + A11 * b2) / det
        
        # Flow 정규화 및 클리핑
        flow = torch.cat([u, v], dim=1)
        flow = torch.clamp(flow, -10, 10)  # 극단적인 값 방지
        
        return flow

class SimpleWarp(nn.Module):
    """개선된 이미지 warping"""
    def __init__(self):
        super(SimpleWarp, self).__init__()
        
    def forward(self, image, flow):
        """
        flow를 사용해서 이미지를 warp (정확한 구현)
        image: (B, C, H, W)
        flow: (B, 2, H, W)
        """
        B, C, H, W = image.size()
        
        # 그리드 생성
        grid_y, grid_x = torch.meshgrid(torch.arange(H), torch.arange(W))
        grid = torch.stack((grid_x, grid_y), dim=0).float().to(image.device)
        grid = grid.unsqueeze(0).repeat(B, 1, 1, 1)
        
        # Flow 정규화 (-1 ~ 1 범위로)
        flow_normalized = flow.clone()
        flow_normalized[:, 0] = flow_normalized[:, 0] / (W - 1) * 2 - 1
        flow_normalized[:, 1] = flow_normalized[:, 1] / (H - 1) * 2 - 1
        
        # 그리드에 flow 적용
        grid = grid + flow_normalized
        
        # grid_sample을 사용한 정확한 warping
        warped = F.grid_sample(image, grid.permute(0, 2, 3, 1), 
                              mode='bilinear', padding_mode='border', align_corners=True)
        
        return warped

class SimpleFlowWrapper:
    """간단한 Optical Flow 래퍼 클래스"""
    def __init__(self):
        self._init_flow()
    
    def _init_flow(self):
        """간단한 Optical Flow 초기화"""
        self.flow_net = SimpleOpticalFlow().cuda()
        self.warp_net = SimpleWarp().cuda()
        print("Simple Optical Flow initialized")
    
    def compute_flow(self, img1, img2):
        """간단한 Optical Flow 계산"""
        with torch.no_grad():
            flow = self.flow_net(img1, img2)
        return flow
    
    def warp_image(self, image, flow):
        """이미지 warping"""
        return self.warp_net(image, flow)

def compute_optical_flow_legacy(img1, img2):
    """FlowNet2를 사용한 광학 흐름 계산"""
    # 전역 Simple Flow 인스턴스
    flow_net = SimpleFlowWrapper()
    return SimpleFlowWrapper.compute_flow(img1, img2)

def warp_with_flow(image, flow):
    """광학 흐름을 사용하여 이미지 워핑"""
    B, C, H, W = image.size()
    
    # 그리드 생성
    grid_y, grid_x = torch.meshgrid(torch.arange(H), torch.arange(W))
    grid = torch.stack((grid_x, grid_y), dim=0).float().to(image.device)
    grid = grid.unsqueeze(0).repeat(B, 1, 1, 1)
    
    # 플로우 적용
    flow_normalized = flow.clone()
    flow_normalized[:, 0] = flow_normalized[:, 0] / (W - 1) * 2 - 1
    flow_normalized[:, 1] = flow_normalized[:, 1] / (H - 1) * 2 - 1
    
    grid = grid + flow_normalized
    
    # 워핑
    warped = F.grid_sample(image, grid.permute(0, 2, 3, 1), 
                          mode='bilinear', padding_mode='border', align_corners=True)
    return warped


def warp_mask_with_flow(mask: torch.Tensor, flow: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """마스크를 flow로 워핑하고 이진화하여 반환.
    - mask: (B, 1, H, W) float
    - flow: (B, 2, H, W) float
    반환: (B, 1, H, W) float {0,1}
    """
    warped = warp_with_flow(mask, flow)
    return (warped > threshold).float()


# --- Functions copied from compute_flow_warp.py ---

# load_image function is not needed as inputs are already tensors
# def load_image(image_path, target_size=(256, 256)):
#     # ... (original implementation not copied as it's for file loading)

def _warp_with_flow(tensor: torch.Tensor,
                    flow,
                    mode: str = "bilinear",
                    padding_mode: str = "border",
                    align_corners: bool = True) -> torch.Tensor:
    """
    grid_sample 규약에 맞게 tensor를 optical flow로 warp.
    - tensor: (B, C, H, W)
    - flow:   (B, 2, H, W) 또는 (B, 2, 1, 1), 픽셀 단위 (우측/하방 +)
    반환: warped (B, C, H, W)
    """
    if isinstance(flow, np.ndarray):
        flow = torch.from_numpy(flow).float()
        if tensor.is_cuda:
            flow = flow.cuda()

    flow = flow.to(tensor.dtype).to(tensor.device)

    B, C, H, W = tensor.shape

    # flow 크기 브로드캐스트 / resize
    if flow.shape[-2:] != (H, W):
        if flow.shape[-2:] == (1, 1):   # 상수 flow
            flow = flow.expand(B, 2, H, W)
        else:
            flow = F.interpolate(flow, size=(H, W), mode='bilinear', align_corners=True)

    # base grid 생성 [-1,1]
    ys = torch.linspace(-1.0, 1.0, H, device=tensor.device, dtype=tensor.dtype)
    xs = torch.linspace(-1.0, 1.0, W, device=tensor.device, dtype=tensor.dtype)
    base_y, base_x = torch.meshgrid(ys, xs, indexing='ij')
    base_grid = torch.stack((base_x, base_y), dim=-1)      # (H, W, 2)
    base_grid = base_grid.unsqueeze(0).expand(B, -1, -1, -1)  # (B, H, W, 2)

    # 픽셀 flow -> 정규화된 delta
    dx = flow[:, 0]    # (B,H,W)
    dy = flow[:, 1]
    dx_norm = (2.0 * dx) / max(W - 1, 1)
    dy_norm = (2.0 * dy) / max(H - 1, 1)
    flow_norm = torch.stack((dx_norm, dy_norm), dim=-1)    # (B,H,W,2)

    # t-1 -> t
    grid = base_grid - flow_norm

    warped = F.grid_sample(
        tensor, grid,
        mode=mode,
        padding_mode=padding_mode,
        align_corners=align_corners
    )
    return warped

def warp_image_with_flow(image: torch.Tensor, flow) -> torch.Tensor:
    """
    - image: (B, C, H, W), 보통 0~1 범위
    - flow:  (B, 2, H, W) or (B,2,1,1)
    반환: warped image (B, C, H, W)
    """
    return _warp_with_flow(image, flow, mode="bilinear", padding_mode="border")

def warp_mask_with_flow(mask: torch.Tensor, flow, thresh: float = 0.5) -> torch.Tensor:
    """
    - mask: (B, 1, H, W), 0~1
    반환: 이진 마스크 (B,1,H,W)
    """
    warped = _warp_with_flow(mask, flow, mode="bilinear", padding_mode="border")
    warped_binary = (warped > thresh).float()
    return warped_binary

def warp_feature_with_flow(feat: torch.Tensor, flow) -> torch.Tensor:
    """
    - feat: (B, C, H, W), 임의 feature map
    - flow: (B, 2, H, W) or (B,2,1,1)
    반환: warped feature (B, C, H, W)
    """
    return _warp_with_flow(feat, flow, mode="bilinear", padding_mode="border")

def compute_optical_flow(img1, img2, modality='rgb', method='farneback', method_kwargs=None):
    """
    다양한 의료 영상에 대응하는 optical flow 계산
    
    Args:
        modality: 'ultrasound' 또는 'rgb' (endoscopy, polyp 등)
    """
    if torch.is_tensor(img1):
        if modality == 'ultrasound':
            # Grayscale 처리
            img1_np = img1.squeeze(1).cpu().numpy() if img1.shape[1] == 1 else img1[:,0].cpu().numpy()
            img2_np = img2.squeeze(1).cpu().numpy() if img2.shape[1] == 1 else img2[:,0].cpu().numpy()
        elif modality == 'rgb':
            # RGB를 grayscale로 변환 (opencv optical flow는 grayscale 필요)
            img1_np = img1.cpu().numpy()  # (B, 3, H, W)
            img2_np = img2.cpu().numpy()
            
            # RGB to grayscale 변환 (luminosity method)
            img1_np = 0.299*img1_np[:,0] + 0.587*img1_np[:,1] + 0.114*img1_np[:,2]
            img2_np = 0.299*img2_np[:,0] + 0.587*img2_np[:,1] + 0.114*img2_np[:,2]
        is_tensor = True
    else:
        img1_np = img1
        img2_np = img2
        is_tensor = False
    
    # Ensure batch dimension for processing
    if img1_np.ndim == 2: # (H, W) -> (1, H, W)
        img1_np = img1_np[np.newaxis, ...]
        img2_np = img2_np[np.newaxis, ...]
    elif img1_np.ndim == 3: # (B, H, W)
        pass # Already in correct format
    else:
        raise ValueError(f"Unsupported image dimensions for flow computation: {img1_np.shape}")
    
    batch_size = img1_np.shape[0]
    flows = []
    
    for b in range(batch_size):
        img1_single = img1_np[b]
        img2_single = img2_np[b]
        
        # 모달리티별 전처리
        if modality == 'ultrasound':
            img1_proc = preprocess_ultrasound_for_flow(img1_single)
            img2_proc = preprocess_ultrasound_for_flow(img2_single)
        elif modality == 'rgb':
            img1_proc = preprocess_rgb_for_flow(img1_single)
            img2_proc = preprocess_rgb_for_flow(img2_single)
        
        if method == 'lucas_kanade':
            flow = compute_lucas_kanade_flow(img1_proc, img2_proc)
        elif method == 'horn_schunck':
            flow = compute_horn_schunck_flow(img1_proc, img2_proc)
        elif method == 'farneback':
            flow = compute_farneback_flow(img1_proc, img2_proc)
        elif method == 'ncc':
            kwargs = method_kwargs or {}
            block_size = int(kwargs.get('block_size', 21))
            search_radius = int(kwargs.get('search_radius', 7))
            stride = int(kwargs.get('stride', 4))
            flow = compute_ncc_block_flow(img1_proc, img2_proc, block_size=block_size, search_radius=search_radius, stride=stride)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        flows.append(flow)
    
    # 결과를 PyTorch 텐서로 변환
    flow_array = np.stack(flows, axis=0)  # (B, 2, H, W)
    
    if is_tensor:
        flow_tensor = torch.from_numpy(flow_array).float()
        if img1.is_cuda:
            flow_tensor = flow_tensor.cuda()
        return flow_tensor
    else:
        return flow_array

def preprocess_rgb_for_flow(img):
    """대장내시경 이미지 전처리"""

    img_uint8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    img_filtered = cv2.bilateralFilter(img_uint8, d=5, sigmaColor=50, sigmaSpace=50)
    
    # 대비 향상 (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img_enhanced = clahe.apply(img_filtered)

    img_smooth = cv2.GaussianBlur(img_enhanced, (3, 3), 0.3)
    
    return img_smooth.astype(np.float32) / 255.0

def preprocess_ultrasound_for_flow(img):
    """
    초음파 이미지를 optical flow 계산에 최적화
    """
    # 1. 노이즈 제거 (bilateral filter - 엣지 보존)
    img_uint8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    img_denoised = cv2.bilateralFilter(img_uint8, d=9, sigmaColor=75, sigmaSpace=75)
    
    # 2. 대비 향상 (CLAHE - Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_enhanced = clahe.apply(img_denoised)
    
    # 3. 가우시안 피라미드로 멀티스케일 처리 준비
    img_smooth = cv2.GaussianBlur(img_enhanced, (3, 3), 0.5) # 0~1
    
    return img_smooth.astype(np.float32) / 255.0

def compute_ncc_block_flow(img1, img2, block_size=21, search_radius=7, stride=4):
    """
    Speckle tracking: NCC 기반 블록 매칭으로 dense optical flow 근사 계산
    - img1, img2: float32 [0,1], shape (H, W), 단일 채널
    - block_size: 템플릿 블록 크기 (홀수)
    - search_radius: 탐색 반경(픽셀)
    - stride: 그리드 간격(픽셀)
    반환: (2, H, W) [u,v] 픽셀 단위
    """
    H, W = img1.shape
    half_b = block_size // 2
    margin = half_b + search_radius

    ys = list(range(margin, H - margin, stride))
    xs = list(range(margin, W - margin, stride))
    if len(ys) < 1 or len(xs) < 1:
        return np.zeros((2, H, W), dtype=np.float32)

    u_coarse = np.zeros((len(ys), len(xs)), dtype=np.float32)
    v_coarse = np.zeros((len(ys), len(xs)), dtype=np.float32)

    for iy, y in enumerate(ys):
        for ix, x in enumerate(xs):
            tmpl = img1[y - half_b:y + half_b + 1, x - half_b:x + half_b + 1]

            y0 = y - half_b - search_radius
            y1 = y + half_b + search_radius + 1
            x0 = x - half_b - search_radius
            x1 = x + half_b + search_radius + 1

            search = img2[y0:y1, x0:x1]
            if search.shape[0] < tmpl.shape[0] or search.shape[1] < tmpl.shape[1]:
                continue

            res = cv2.matchTemplate(search, tmpl, method=cv2.TM_CCOEFF_NORMED)
            _, _, _, max_loc = cv2.minMaxLoc(res)

            match_y = max_loc[1] + half_b
            match_x = max_loc[0] + half_b

            dy = (y0 + match_y) - y
            dx = (x0 + match_x) - x

            u_coarse[iy, ix] = dx
            v_coarse[iy, ix] = dy

    u_dense = cv2.resize(u_coarse, (W, H), interpolation=cv2.INTER_CUBIC)
    v_dense = cv2.resize(v_coarse, (W, H), interpolation=cv2.INTER_CUBIC)

    u_dense = median_filter_flow(u_dense)
    v_dense = median_filter_flow(v_dense)

    return np.stack([u_dense, v_dense], axis=0)

def compute_lucas_kanade_flow(img1, img2, window_size=15):
    """
    Lucas-Kanade 방법으로 optical flow 계산 (초음파 최적화)
    """
    # 그래디언트 계산 (Sobel 필터 사용)
    Ix = cv2.Sobel(img1, cv2.CV_64F, 1, 0, ksize=3) * 0.125
    Iy = cv2.Sobel(img1, cv2.CV_64F, 0, 1, ksize=3) * 0.125
    It = img2 - img1
    
    # 윈도우 크기 설정
    w = window_size // 2
    H, W = img1.shape
    
    # Flow 초기화
    u = np.zeros((H, W))
    v = np.zeros((H, W))
    
    # 각 픽셀에 대해 Lucas-Kanade 계산
    for y in range(w, H - w):
        for x in range(w, W - w):
            # 윈도우 영역 추출
            Ix_win = Ix[y-w:y+w+1, x-w:x+w+1].flatten()
            Iy_win = Iy[y-w:y+w+1, x-w:x+w+1].flatten()
            It_win = It[y-w:y+w+1, x-w:x+w+1].flatten()
            
            # A 행렬과 b 벡터 구성
            A = np.column_stack([Ix_win, Iy_win])
            b = -It_win
            
            # 최소자승법으로 해 구하기 (regularization 추가)
            ATA = A.T @ A
            ATb = A.T @ b
            
            # 조건수 확인 (특이점 처리)
            if np.linalg.det(ATA) > 1e-6:
                # 정규화 항 추가 (Tikhonov regularization)
                ATA += np.eye(2) * 1e-6
                flow_vec = np.linalg.solve(ATA, ATb)
                u[y, x] = flow_vec[0]
                v[y, x] = flow_vec[1]
    
    # Flow 후처리 (이상치 제거)
    u = median_filter_flow(u)
    v = median_filter_flow(v)
    
    return np.stack([u, v], axis=0)

def compute_farneback_flow(img1, img2):
    """
    OpenCV Farneback 방법으로 optical flow 계산 (초음파 최적화)
    """
    # 이미지를 uint8 단일 채널로 변환
    img1_uint8 = (img1 * 255).astype(np.uint8)
    img2_uint8 = (img2 * 255).astype(np.uint8)

    # Farneback dense optical flow 계산
    flow = cv2.calcOpticalFlowFarneback(
        img1_uint8,          # prev
        img2_uint8,          # next
        None,                # initial flow
        0.5,                 # pyr_scale
        3,                   # levels
        15,                  # winsize
        3,                   # iterations
        5,                   # poly_n
        1.2,                 # poly_sigma
        0                    # flags
    )

    # Flow 후처리 (이상치 제거)
    flow_u = median_filter_flow(flow[:, :, 0])
    flow_v = median_filter_flow(flow[:, :, 1])

    return np.stack([flow_u, flow_v], axis=0)

def compute_horn_schunck_flow(img1, img2, alpha=0.1, iterations=100):
    """
    Horn-Schunck 방법으로 optical flow 계산
    """
    # 그래디언트 계산
    Ix = cv2.Sobel(img1, cv2.CV_64F, 1, 0, ksize=3) * 0.125
    Iy = cv2.Sobel(img1, cv2.CV_64F, 0, 1, ksize=3) * 0.125
    It = img2 - img1
    
    # Flow 초기화
    u = np.zeros_like(img1)
    v = np.zeros_like(img1)
    
    # 반복적 해법
    for _ in range(iterations):
        # 평균 필터 (라플라시안 근사)
        u_avg = cv2.GaussianBlur(u, (3, 3), 0)
        v_avg = cv2.GaussianBlur(v, (3, 3), 0)
        
        # Horn-Schunck 업데이트
        denominator = alpha**2 + Ix**2 + Iy**2 + 1e-8
        numerator = Ix * u_avg + Iy * v_avg + It
        
        u = u_avg - Ix * numerator / denominator
        v = v_avg - Iy * numerator / denominator
    
    return np.stack([u, v], axis=0)

def median_filter_flow(flow, kernel_size=3):
    """
    Flow에 median filter 적용하여 이상치 제거
    """
    return ndimage.median_filter(flow, size=kernel_size)

# --- End of functions copied from compute_flow_warp.py ---

# Re-implement compute_temporal_consistency_metric to use the new warping
def compute_temporal_consistency_metric(pred_sequence, flow_sequence=None):
    """
    시간적 일관성 메트릭 계산
    """
    consistency_scores = []
    
    for i in range(1, len(pred_sequence)):
        current_pred = pred_sequence[i]
        previous_pred = pred_sequence[i-1]
        
        if flow_sequence is not None and i-1 < len(flow_sequence):
            flow = flow_sequence[i-1]
            # Use the new warp_image_with_flow
            warped_prev = warp_image_with_flow(previous_pred, flow)
            consistency = 1 - F.mse_loss(current_pred, warped_prev)
        else:
            consistency = 1 - F.mse_loss(current_pred, previous_pred)
            
        consistency_scores.append(consistency.item())
    
    return np.mean(consistency_scores)


def generate_flip_pseudo_label(model, images):
    """
    Flip-based Pseudo Label 생성
    원본 이미지와 좌우 반전된 이미지를 모두 모델에 입력하여 더 정확한 pseudo label 생성
    """
    batch_size, sequence_length = images.shape[:2]
    pseudo_labels = []
    
    for seq_idx in range(sequence_length):
        img_batch = images[:, seq_idx].cuda()  # (B, C, H, W)
        
        # 원본 이미지 예측
        with torch.no_grad():
            outputs_original = model(img_batch)
            masks_original = torch.sigmoid(outputs_original)  # (B, 1, H, W)
        
        # 좌우 반전된 이미지 예측
        img_flipped = torch.flip(img_batch, dims=[3])  # 좌우 반전
        with torch.no_grad():
            outputs_flipped = model(img_flipped)
            masks_flipped = torch.sigmoid(outputs_flipped)  # (B, 1, H, W)
        
        # 좌우 반전된 마스크를 다시 원래대로 뒤집기
        pseudo_mask = torch.flip(masks_flipped, dims=[3])  # 다시 좌우 반전
        
        # 원본 마스크와 뒤집힌 마스크의 평균 (더 안정적인 pseudo label)
        #pseudo_mask = (masks_original + masks_flipped_back) / 2.0
        
        pseudo_labels.append(pseudo_mask)
    
    # 텐서로 변환
    pseudo_labels = torch.stack(pseudo_labels, dim=1)  # (B, sequence_length, 1, H, W)
    return pseudo_labels



def _fmt_float(x: float) -> str:
    """1.5 -> 1p5, 0.001 -> 0p001 (파일명 안전 포맷)"""
    return f"{x}".replace(".", "p")


def build_run_tag(args) -> str:
    """
    간결 태그: 모델, view, rotate 표시, λ들, flow, (선택) seed
    예) U_Net_view-axi_rot_seg1p5_con1p5_kd3p0_flow-farneback_seed41
    """
    parts = [
        args.model,
        f"view-{args.view_type}",
        "rot",  # 회전 기반 pseudo-label 사용 표시
        f"seg{_fmt_float(args.seg_lam)}",
        f"con{_fmt_float(args.con_lam)}",
        f"kd{_fmt_float(args.kd_lam)}",
        f"flow-{args.flow_method}",
        f"seed{args.seed}",  # 필요 없으면 이 줄 삭제해도 됨
    ]
    return "_".join(parts)


def build_rotation_matrices(angles_deg: torch.Tensor) -> torch.Tensor:
    """Create batch 2x3 rotation matrices for given angles in degrees."""
    angles_rad = angles_deg * (np.pi / 180.0)
    cos_a = torch.cos(angles_rad)
    sin_a = torch.sin(angles_rad)
    zeros = torch.zeros_like(cos_a)
    row1 = torch.stack([cos_a, -sin_a, zeros], dim=1)
    row2 = torch.stack([sin_a,  cos_a, zeros], dim=1)
    theta = torch.stack([row1, row2], dim=1)  # (B, 2, 3)
    return theta


def rotate_batch(images: torch.Tensor, angles_deg: torch.Tensor) -> torch.Tensor:
    """Rotate a batch of images by per-sample angles (degrees). images: (B,C,H,W)."""
    theta = build_rotation_matrices(angles_deg.to(images.device).to(images.dtype))
    grid = F.affine_grid(theta, size=images.size(), align_corners=True)
    rotated = F.grid_sample(images, grid, mode='bilinear', padding_mode='zeros', align_corners=True)
    return rotated




def knowledge_distillation_loss(student_pred, teacher_pred, temperature=4.0):
    """
    Knowledge Distillation Loss 계산
    student_pred: 학습 중인 모델의 예측 (logits)
    teacher_pred: teacher 모델의 예측 (logits)
    temperature: distillation temperature
    """
    # Temperature scaling
    student_logits = student_pred / temperature
    teacher_logits = teacher_pred / temperature

    # Soft targets (teacher의 softmax)
    teacher_probs = torch.sigmoid(teacher_logits)

    # Knowledge distillation loss (KL divergence)
    student_probs = torch.sigmoid(student_logits)
    kd_loss = F.binary_cross_entropy(student_probs, teacher_probs)

    return kd_loss


def _apply_brightness_contrast(img_batch: torch.Tensor, brightness: float, contrast: float) -> torch.Tensor:
    b = img_batch.shape[0]
    device = img_batch.device
    dtype = img_batch.dtype
    out = img_batch
    if brightness > 0:
        factors = 1.0 + (torch.rand(b, 1, 1, 1, device=device, dtype=dtype) * (2 * brightness) - brightness)
        out = out * factors
    if contrast > 0:
        means = out.mean(dim=(2, 3), keepdim=True)
        factors = 1.0 + (torch.rand(b, 1, 1, 1, device=device, dtype=dtype) * (2 * contrast) - contrast)
        out = (out - means) * factors + means
    return out


def _apply_gaussian_noise(img_batch: torch.Tensor, std: float) -> torch.Tensor:
    if std <= 0:
        return img_batch
    noise = torch.randn_like(img_batch) * std
    return (img_batch + noise)


def _maybe_blur(img_batch: torch.Tensor, kernel_size: int, p: float) -> torch.Tensor:
    if p <= 0:
        return img_batch
    if float(torch.rand(1)) >= p:
        return img_batch
    k = max(1, int(kernel_size))
    if k % 2 == 0:
        k += 1
    pad = k // 2
    # 간단한 평균 블러로 근사
    out = torch.nn.functional.avg_pool2d(img_batch, kernel_size=k, stride=1, padding=pad)
    return out


def _coarse_dropout(img_batch: torch.Tensor, max_holes: int = 4, max_ratio: float = 0.1, p: float = 0.0) -> torch.Tensor:
    if p <= 0 or max_holes <= 0 or max_ratio <= 0:
        return img_batch
    if float(torch.rand(1)) >= p:
        return img_batch
    b, c, h, w = img_batch.shape
    out = img_batch.clone()
    for bi in range(b):
        num_holes = int(torch.randint(1, max_holes + 1, (1,)).item())
        for _ in range(num_holes):
            hole_h = int(max(1, h * (torch.rand(1).item() * max_ratio)))
            hole_w = int(max(1, w * (torch.rand(1).item() * max_ratio)))
            y0 = int(torch.randint(0, max(1, h - hole_h), (1,)).item())
            x0 = int(torch.randint(0, max(1, w - hole_w), (1,)).item())
            out[bi, :, y0:y0 + hole_h, x0:x0 + hole_w] = 0.0
    return out


def apply_teacher_weak_augmentations(img_batch: torch.Tensor) -> torch.Tensor:
    """
    Teacher용 약한 포토메트릭 증강(기하학 없음, 정렬 유지)
    """
    out = _apply_brightness_contrast(img_batch, brightness=0.1, contrast=0.1)
    out = _apply_gaussian_noise(out, std=0.01)
    out = _maybe_blur(out, kernel_size=3, p=0.1)
    return out.clamp(0.0, 1.0)


def apply_student_strong_augmentations(img_batch: torch.Tensor) -> torch.Tensor:
    """
    Student용 강한 포토메트릭 증강(기하학 없음, 정렬 유지)
    """
    out = _apply_brightness_contrast(img_batch, brightness=0.3, contrast=0.3)
    out = _apply_gaussian_noise(out, std=0.08)
    out = _maybe_blur(out, kernel_size=5, p=0.4)
    out = _coarse_dropout(out, max_holes=4, max_ratio=0.12, p=0.3)
    return out.clamp(0.0, 1.0)
