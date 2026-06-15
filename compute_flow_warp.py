import os
import torch
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from PIL import Image
import argparse
from src.utils.util import SimpleOpticalFlow, SimpleWarp, SimpleFlowWrapper, compute_optical_flow, warp_with_flow
from scipy import ndimage

def load_image(image_path, target_size=(256, 256)):
    """
    이미지 로드 및 전처리 (그레이스케일 초음파 이미지 최적화)
    """
    # PIL로 이미지 로드 (그레이스케일로)
    img = Image.open(image_path).convert('L')  # 그레이스케일로 변환
    
    # 크기 조정
    img = img.resize(target_size, Image.LANCZOS)
    
    # numpy로 변환
    img_np = np.array(img).astype(np.float32) / 255.0
    
    # 초음파 이미지 전처리
    # 1. 노이즈 제거 (가우시안 블러)
    img_blur = cv2.GaussianBlur(img_np, (3, 3), 0)
    
    # 2. 대비 향상 (히스토그램 평활화)
    img_enhanced = cv2.equalizeHist((img_blur * 255).astype(np.uint8)).astype(np.float32) / 255.0
    
    # 3. 3채널로 확장 (flow 계산을 위해)
    img_processed = np.stack([img_enhanced, img_enhanced, img_enhanced], axis=2)
    
    # PyTorch 텐서로 변환 (B, C, H, W)
    img_tensor = torch.from_numpy(img_processed).permute(2, 0, 1).unsqueeze(0)
    
    return img_tensor, img_processed

def warp_image_with_flow(image: torch.Tensor, flow) -> torch.Tensor:
    """
    Flow를 사용해 이미지를 warp. grid_sample 규약에 맞게 정확히 구현.
    - image: (B, C, H, W), 0~1
    - flow: (B, 2, H, W) 또는 (B, 2, 1, 1) 픽셀 단위(우측/하방 +)
    반환: warped (B, C, H, W)
    """
    if isinstance(flow, np.ndarray):
        flow = torch.from_numpy(flow).float()
        if image.is_cuda:
            flow = flow.cuda()
    flow = flow.to(image.dtype).to(image.device)

    B, C, H, W = image.shape
    # flow 크기 브로드캐스트
    if flow.shape[-2:] != (H, W):
        # 상수 flow라면 브로드캐스트
        if flow.shape[-2:] == (1, 1):
            flow = flow.expand(B, 2, H, W)
        else:
            flow = torch.nn.functional.interpolate(flow, size=(H, W), mode='bilinear', align_corners=True)

    # 정규화 그리드 생성 [-1, 1]
    ys = torch.linspace(-1.0, 1.0, H, device=image.device, dtype=image.dtype)
    xs = torch.linspace(-1.0, 1.0, W, device=image.device, dtype=image.dtype)
    base_y, base_x = torch.meshgrid(ys, xs, indexing='ij') if hasattr(torch, 'meshgrid') else torch.meshgrid(ys, xs)
    base_grid = torch.stack((base_x, base_y), dim=-1)  # (H, W, 2)
    base_grid = base_grid.unsqueeze(0).repeat(B, 1, 1, 1)  # (B, H, W, 2)

    # 픽셀 flow를 정규화 델타로 변환
    # dx_norm = 2*dx/(W-1), dy_norm = 2*dy/(H-1)
    dx = flow[:, 0, :, :]
    dy = flow[:, 1, :, :]
    dx_norm = (2.0 * dx) / max(W - 1, 1)
    dy_norm = (2.0 * dy) / max(H - 1, 1)
    flow_norm = torch.stack((dx_norm, dy_norm), dim=-1)  # (B, H, W, 2)

    grid = base_grid + flow_norm
    warped = torch.nn.functional.grid_sample(
        image, grid, mode='bilinear', padding_mode='border', align_corners=True
    )
    return warped

def compute_ultrasound_optical_flow(img1, img2, method='lucas_kanade', method_kwargs=None):
    """
    그레이스케일 초음파 이미지에 특화된 optical flow 계산
    
    Args:
        img1, img2: (B, C, H, W) 형태의 PyTorch 텐서 또는 (H, W) numpy 배열
        method: 'lucas_kanade', 'horn_schunck', 'farneback' 중 선택
    
    Returns:
        flow: (B, 2, H, W) 형태의 optical flow
    """
    # PyTorch 텐서를 numpy로 변환
    if torch.is_tensor(img1):
        img1_np = img1.squeeze().cpu().numpy()
        img2_np = img2.squeeze().cpu().numpy()
        is_tensor = True
    else:
        img1_np = img1
        img2_np = img2
        is_tensor = False
    
    # 배치 차원 확인
    if len(img1_np.shape) == 4:  # (B, C, H, W)
        batch_size = img1_np.shape[0]
        img1_gray = img1_np[:, 0] if img1_np.shape[1] == 3 else img1_np.squeeze(1)
        img2_gray = img2_np[:, 0] if img2_np.shape[1] == 3 else img2_np.squeeze(1)
    elif len(img1_np.shape) == 3:  # (C, H, W)
        batch_size = 1
        img1_gray = img1_np[0] if img1_np.shape[0] == 3 else img1_np.squeeze(0)
        img2_gray = img2_np[0] if img2_np.shape[0] == 3 else img2_np.squeeze(0)
        img1_gray = img1_gray[np.newaxis, ...]
        img2_gray = img2_gray[np.newaxis, ...]
    else:  # (H, W)
        batch_size = 1
        img1_gray = img1_np[np.newaxis, ...]
        img2_gray = img2_np[np.newaxis, ...]
    
    flows = []
    
    for b in range(batch_size):
        img1_single = img1_gray[b]
        img2_single = img2_gray[b]
        
        # 이미지 전처리 (초음파 특화)
        img1_proc = preprocess_ultrasound_for_flow(img1_single)
        img2_proc = preprocess_ultrasound_for_flow(img2_single)
        
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

def preprocess_ultrasound_for_flow(img):
    """
    초음파 이미지를 optical flow 계산에 최적화
    """
    # 1. 노이즈 제거 (bilateral filter - 엣지 보존)
    img_denoised = cv2.bilateralFilter((img * 255).astype(np.uint8), 9, 75, 75).astype(np.float32) / 255.0
    
    # 2. 대비 향상 (CLAHE - Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_enhanced = clahe.apply((img_denoised * 255).astype(np.uint8)).astype(np.float32) / 255.0
    
    # 3. 가우시안 피라미드로 멀티스케일 처리 준비
    img_smooth = cv2.GaussianBlur(img_enhanced, (3, 3), 0.5)
    
    return img_smooth

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
    # 참고: https://docs.opencv.org/4.x/d4/dee/tutorial_optical_flow.html
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

def compute_flow_and_warp(img1_path, img2_path, output_dir="flow_warp_results",
                          arrow_stride=8, arrow_min_mag=0.0, arrow_scale=1.0, arrow_normalize=True,
                          save_each_method=True, swap_xy=False, invert_y=False):
    """
    두 이미지 간의 optical flow와 warping 계산
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 이미지 로드
    print(f"Loading images...")
    img1_tensor, img1_np = load_image(img1_path)
    img2_tensor, img2_np = load_image(img2_path)
    
    # GPU로 이동
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    img1_tensor = img1_tensor.to(device)
    img2_tensor = img2_tensor.to(device)
    
    print(f"Computing optical flow...")
    
    # 1. 기존 Optical Flow 계산 (img1 -> img2)
    print("=== Original Flow Method ===")
    flow_original = compute_optical_flow(img1_tensor, img2_tensor)  # img1 -> img2 방향
    print(f"Original flow - min: {flow_original.min():.6f}, max: {flow_original.max():.6f}, mean: {flow_original.mean():.6f}")
    
    # 2. 새로운 초음파 특화 Optical Flow 계산
    print("=== Ultrasound-Optimized Flow Methods ===")
    
    # Lucas-Kanade 방법
    print("Computing Lucas-Kanade flow...")
    flow_lk = compute_ultrasound_optical_flow(img1_tensor, img2_tensor, method='lucas_kanade')
    print(f"Lucas-Kanade flow - min: {flow_lk.min():.6f}, max: {flow_lk.max():.6f}, mean: {flow_lk.mean():.6f}")
    
    # Farneback 방법
    print("Computing Farneback flow...")
    flow_farneback = compute_ultrasound_optical_flow(img1_tensor, img2_tensor, method='farneback')
    print(f"Farneback flow - min: {flow_farneback.min():.6f}, max: {flow_farneback.max():.6f}, mean: {flow_farneback.mean():.6f}")
    
    # Horn-Schunck 방법
    print("Computing Horn-Schunck flow...")
    flow_hs = compute_ultrasound_optical_flow(img1_tensor, img2_tensor, method='horn_schunck')
    print(f"Horn-Schunck flow - min: {flow_hs.min():.6f}, max: {flow_hs.max():.6f}, mean: {flow_hs.mean():.6f}")

    # NCC Speckle Tracking
    print("Computing NCC (speckle tracking) flow...")
    flow_ncc = compute_ultrasound_optical_flow(
        img1_tensor, img2_tensor, method='ncc',
        method_kwargs={'block_size': 21, 'search_radius': 7, 'stride': 4}
    )
    print(f"NCC flow - min: {flow_ncc.min():.6f}, max: {flow_ncc.max():.6f}, mean: {flow_ncc.mean():.6f}")
    
    # 3. Warping 계산 (각 방법별로)
    flow_net = SimpleFlowWrapper()
    warp_original = warp_image_with_flow(img1_tensor, flow_original)
    warp_lk = warp_image_with_flow(img1_tensor, flow_lk)
    warp_farneback = warp_image_with_flow(img1_tensor, flow_farneback)
    warp_hs = warp_image_with_flow(img1_tensor, flow_hs)
    warp_ncc = warp_image_with_flow(img1_tensor, flow_ncc)
    
    # 가장 좋은 flow 선택 (magnitude가 가장 큰 것)
    flows = {
        'original': flow_original,
        'lucas_kanade': flow_lk,
        'farneback': flow_farneback,
        'horn_schunck': flow_hs,
        'ncc': flow_ncc
    }
    
    flow_magnitudes = {}
    for name, f in flows.items():
        mag = torch.sqrt(f[:, 0]**2 + f[:, 1]**2).mean().item()
        flow_magnitudes[name] = mag
        print(f"{name} average magnitude: {mag:.6f}")
    
    # 가장 큰 magnitude를 가진 flow 선택 (기본)
    best_method = max(flow_magnitudes, key=flow_magnitudes.get)
    flow = flows[best_method]
    warp_img1_to_img2 = warp_image_with_flow(img1_tensor, flow)
    
    print(f"Selected best method: {best_method} (magnitude: {flow_magnitudes[best_method]:.6f})")
    
    # 4. NOC (Non-Occlusion) 마스크 계산
    img_diff = torch.abs(torch.sum(img2_tensor - warp_img1_to_img2, dim=1, keepdim=True))
    noc_mask = torch.exp(-1 * img_diff)
    
    # CPU로 이동하여 numpy 변환
    img1_np = img1_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    img2_np = img2_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    flow_np = flow.squeeze(0).permute(1, 2, 0).cpu().numpy()
    warp_img1_to_img2_np = warp_img1_to_img2.squeeze(0).permute(1, 2, 0).cpu().numpy()
    noc_mask_np = noc_mask.squeeze(0).squeeze(0).cpu().numpy()
    
    # 5. 결과 저장
    print(f"Saving results to {output_dir}...")
    
    # 원본 이미지 저장
    plt.imsave(os.path.join(output_dir, 'img1_original.png'), img1_np)
    plt.imsave(os.path.join(output_dir, 'img2_original.png'), img2_np)
    
    # Warped 이미지 저장
    plt.imsave(os.path.join(output_dir, 'img1_warped_to_img2.png'), np.clip(warp_img1_to_img2_np, 0, 1))

    # 방법별 Warped 이미지도 저장 (디버깅/비교용)
    if save_each_method:
        plt.imsave(os.path.join(output_dir, 'warp_original.png'),
                   np.clip(warp_original.squeeze(0).permute(1, 2, 0).detach().cpu().numpy(), 0, 1))
        plt.imsave(os.path.join(output_dir, 'warp_lucas_kanade.png'),
                   np.clip(warp_lk.squeeze(0).permute(1, 2, 0).detach().cpu().numpy(), 0, 1))
        plt.imsave(os.path.join(output_dir, 'warp_farneback.png'),
                   np.clip(warp_farneback.squeeze(0).permute(1, 2, 0).detach().cpu().numpy(), 0, 1))
        plt.imsave(os.path.join(output_dir, 'warp_horn_schunck.png'),
                   np.clip(warp_hs.squeeze(0).permute(1, 2, 0).detach().cpu().numpy(), 0, 1))
    
    # NOC 마스크 저장
    plt.imsave(os.path.join(output_dir, 'noc_mask.png'), noc_mask_np, cmap='gray')
    
    # 6. Optical Flow 시각화
    def flow_to_rgb(flow, swap_xy_local=False):
        """Optical flow를 RGB 이미지로 변환"""
        # Flow의 크기와 방향 계산
        fx = flow[:, :, 0]
        fy = flow[:, :, 1]
        if swap_xy_local:
            fx, fy = fy, fx
        magnitude = np.sqrt(fx**2 + fy**2)
        angle = np.arctan2(fy, fx)
        
        # 디버깅 정보 출력
        print(f"Flow magnitude - min: {magnitude.min():.6f}, max: {magnitude.max():.6f}, mean: {magnitude.mean():.6f}")
        print(f"Flow angle - min: {angle.min():.6f}, max: {angle.max():.6f}, mean: {angle.mean():.6f}")
        
        # HSV 색상 공간 (matplotlib.colors.hsv_to_rgb 사용: 모든 채널 0~1 범위)
        h = (angle + np.pi) / (2 * np.pi)  # 0~1
        magnitude_normalized = magnitude / (magnitude.max() + 1e-8)
        s = np.clip(magnitude_normalized, 0.1, 1.0)
        v = np.clip(0.5 + 0.5 * magnitude_normalized, 0.3, 1.0)

        hsv = np.stack([h, s, v], axis=2)
        rgb = mcolors.hsv_to_rgb(hsv)
        
        return rgb
    
    # Flow 시각화 및 저장
    flow_rgb = flow_to_rgb(flow_np, swap_xy_local=swap_xy)
    
    plt.imsave(os.path.join(output_dir, 'flow_img1_to_img2.png'), flow_rgb)
    
    # 추가: Flow magnitude 직접 시각화 (그레이스케일 초음파용)
    magnitude = np.sqrt(flow_np[:, :, 0]**2 + flow_np[:, :, 1]**2)
    
    # 마스크 영역에서만 flow 계산 (0,1 픽셀값 변화가 큰 영역)
    # magnitude가 특정 임계값 이상인 영역만 강조
    magnitude_threshold = magnitude > (magnitude.mean() + 2 * magnitude.std())
    
    plt.figure(figsize=(10, 10))
    plt.imshow(magnitude, cmap='hot', vmin=0, vmax=magnitude.max())
    plt.colorbar(label='Flow Magnitude')
    plt.title('Optical Flow Magnitude (Gray-scale Ultrasound)')
    plt.axis('off')
    plt.savefig(os.path.join(output_dir, 'flow_magnitude.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # Flow 방향 시각화 (마스크 변화가 큰 영역 강조)
    angle = np.arctan2(flow_np[:, :, 1], flow_np[:, :, 0])
    
    plt.figure(figsize=(10, 10))
    plt.imshow(angle, cmap='hsv', vmin=-np.pi, vmax=np.pi)
    plt.colorbar(label='Flow Direction (radians)')
    plt.title('Optical Flow Direction (Gray-scale Ultrasound)')
    plt.axis('off')
    plt.savefig(os.path.join(output_dir, 'flow_direction.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # Flow magnitude 히스토그램 (분포 확인)
    plt.figure(figsize=(8, 6))
    plt.hist(magnitude.flatten(), bins=50, alpha=0.7, color='blue')
    plt.axvline(magnitude.mean(), color='red', linestyle='--', label=f'Mean: {magnitude.mean():.6f}')
    plt.axvline(magnitude.mean() + 2*magnitude.std(), color='orange', linestyle='--', label=f'Mean+2*Std: {magnitude.mean() + 2*magnitude.std():.6f}')
    plt.xlabel('Flow Magnitude')
    plt.ylabel('Frequency')
    plt.title('Flow Magnitude Distribution')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'flow_magnitude_histogram.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # OpenCV 기반 픽셀 좌표계 화살표 오버레이 함수 (사용 전에 정의)
    def draw_flow_arrows_cv2(flow, img_gray01, out_path, step=8, scale=3.0, swap_xy=False, invert_y=False):
        h, w = flow.shape[:2]
        # 배경: 0~1 그레이 → 0~255 BGR
        if img_gray01.ndim == 3:
            bg = (img_gray01[:, :, 0] * 255.0).astype(np.uint8)
        else:
            bg = (img_gray01 * 255.0).astype(np.uint8)
        bg_bgr = cv2.cvtColor(bg, cv2.COLOR_GRAY2BGR)

        for y in range(0, h, step):
            for x in range(0, w, step):
                dx = flow[y, x, 0]
                dy = flow[y, x, 1]
                if swap_xy:
                    dx, dy = dy, dx
                if invert_y:
                    dy = -dy
                end_x = int(round(x + dx * scale))
                end_y = int(round(y + dy * scale))
                cv2.arrowedLine(bg_bgr, (x, y), (end_x, end_y), (0, 0, 255), 1, tipLength=0.3)

        cv2.imwrite(out_path, bg_bgr)

    # 7. 모든 방법 비교 시각화 + 각 방법별 화살표 저장
    methods_for_fig = ['original', 'lucas_kanade', 'farneback', 'horn_schunck', 'ncc']
    flows_for_fig = [flows[k] for k in methods_for_fig]
    warps_for_fig = [warp_original, warp_lk, warp_farneback, warp_hs, warp_ncc]
    fig, axes = plt.subplots(3, len(methods_for_fig), figsize=(5*len(methods_for_fig), 15))
    
    # 각 방법별 flow 시각화
    for i, (method, f_data, w_data) in enumerate(zip(methods_for_fig, flows_for_fig, warps_for_fig)):
        # Flow 시각화
        f_np = f_data.squeeze(0).permute(1, 2, 0).cpu().numpy()
        flow_rgb_method = flow_to_rgb(f_np)
        axes[0, i].imshow(flow_rgb_method)
        axes[0, i].set_title(f'Flow: {method}')
        axes[0, i].axis('off')
        
        # Warped 이미지 시각화
        w_np = w_data.squeeze(0).permute(1, 2, 0).cpu().numpy()
        axes[1, i].imshow(np.clip(w_np, 0, 1))
        axes[1, i].set_title(f'Warped: {method}')
        axes[1, i].axis('off')
        
        # Flow magnitude 시각화
        magnitude = np.sqrt(f_np[:, :, 0]**2 + f_np[:, :, 1]**2)
        axes[2, i].imshow(magnitude, cmap='hot')
        axes[2, i].set_title(f'Magnitude: {method}')
        axes[2, i].axis('off')

        # 방법별 화살표 이미지(OpenCV) 저장
        draw_flow_arrows_cv2(f_np, img1_np,
                             os.path.join(output_dir, f'flow_arrows_cv2_{method}.png'),
                             step=arrow_stride, scale=max(arrow_scale, 3.0),
                             swap_xy=swap_xy, invert_y=invert_y)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'flow_methods_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # 8. 원본 종합 시각화 (선택된 최적 방법)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 첫 번째 행
    axes[0, 0].imshow(img1_np)
    axes[0, 0].set_title('Image 1 (Original)')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(img2_np)
    axes[0, 1].set_title('Image 2 (Original)')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(np.clip(warp_img1_to_img2_np, 0, 1))
    axes[0, 2].set_title('Image 1 → Image 2 (Warped)')
    axes[0, 2].axis('off')
    
    # 두 번째 행
    axes[1, 0].imshow(flow_rgb)
    axes[1, 0].set_title('Optical Flow (Image 1 → Image 2)')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(noc_mask_np, cmap='gray')
    axes[1, 1].set_title('NOC Mask (Non-Occlusion)')
    axes[1, 1].axis('off')
    
    # Flow magnitude 히스토그램
    magnitude = np.sqrt(flow_np[:, :, 0]**2 + flow_np[:, :, 1]**2)
    axes[1, 2].hist(magnitude.flatten(), bins=50, alpha=0.7)
    axes[1, 2].set_title('Flow Magnitude Distribution')
    axes[1, 2].set_xlabel('Magnitude')
    axes[1, 2].set_ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'flow_warp_summary.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # 8. 통계 정보 출력
    print(f"\n=== Flow Statistics ===")
    print(f"Flow magnitude - min: {magnitude.min():.4f}, max: {magnitude.max():.4f}, mean: {magnitude.mean():.4f}")
    
    # Flow direction 계산
    angle = np.arctan2(flow_np[:, :, 1], flow_np[:, :, 0])
    print(f"Flow direction - min: {angle.min():.4f}, max: {angle.max():.4f}, mean: {angle.mean():.4f}")
    print(f"NOC mask - min: {noc_mask_np.min():.4f}, max: {noc_mask_np.max():.4f}, mean: {noc_mask_np.mean():.4f}")
    
    # 9. Flow 벡터 필드 시각화
    def plot_flow_vectors(flow, img, output_path, step=8, min_mag=0.0, scale=1.0, normalize=True, swap_xy_local=False, invert_y_local=False):
        """Flow 벡터 필드 시각화"""
        h, w = flow.shape[:2]
        y, x = np.mgrid[0:h:step, 0:w:step].reshape(2, -1)
        fx = flow[y, x, 0]
        fy = flow[y, x, 1]
        if swap_xy_local:
            fx, fy = fy, fx
        mag = np.sqrt(fx**2 + fy**2)

        # 임계값 이하 제거
        keep = mag >= min_mag
        x, y, fx, fy, mag = x[keep], y[keep], fx[keep], fy[keep], mag[keep]

        # 정규화 (방향만 표시하고 싶을 때)
        if normalize:
            fx = np.divide(fx, (mag + 1e-8))
            fy = np.divide(fy, (mag + 1e-8))

        plt.figure(figsize=(10, 10))
        ax = plt.gca()
        plt.imshow(img, cmap='gray')
        # 필요 시 y축 뒤집기 토글
        if invert_y_local:
            ax.invert_yaxis()
        # scale_units='xy', scale가 작을수록 화살표가 길어짐
        plt.quiver(x, y, fx, fy, color='red', scale_units='xy', scale=1.0/scale, width=0.0015,
                   angles='xy', headwidth=2, headlength=3, headaxislength=3, pivot='mid')
        plt.title('Optical Flow Vector Field (Image 1 → Image 2)')
        plt.axis('off')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    # Flow 벡터 필드 저장
    plot_flow_vectors(flow_np, img1_np,
                      os.path.join(output_dir, 'flow_vectors_img1_to_img2.png'),
                      step=arrow_stride, min_mag=arrow_min_mag, scale=arrow_scale,
                      normalize=arrow_normalize, swap_xy_local=swap_xy, invert_y_local=invert_y)

    # 선택된 설정으로 OpenCV 화살표 저장 (swap_xy 반영)
    draw_flow_arrows_cv2(flow_np, img1_np, os.path.join(output_dir, 'flow_arrows_cv2.png'),
                         step=arrow_stride, scale=max(arrow_scale, 3.0), swap_xy=swap_xy, invert_y=invert_y)
    
    print(f"\nResults saved to: {output_dir}")
    print(f"Files created:")
    print(f"- img1_original.png, img2_original.png")
    print(f"- img1_warped_to_img2.png (best method)")
    print(f"- flow_img1_to_img2.png (best method)")
    print(f"- flow_methods_comparison.png (모든 방법 비교)")
    print(f"- flow_magnitude.png (그레이스케일 초음파용)")
    print(f"- flow_direction.png (그레이스케일 초음파용)")
    print(f"- flow_magnitude_histogram.png (분포 확인용)")
    print(f"- noc_mask.png")
    print(f"- flow_warp_summary.png")
    print(f"- flow_vectors_img1_to_img2.png")
    
    return {
        'flow': flow_np,
        'warp_img1_to_img2': warp_img1_to_img2_np,
        'noc_mask': noc_mask_np
    }

def main():
    parser = argparse.ArgumentParser(description='Compute optical flow and warping between two images')
    parser.add_argument('--img1', type=str, default='/workspace/data/axi/masks/0/000503090301_20230418_00018_0000001318.png', help='Path to first image')
    parser.add_argument('--img2', type=str, default='/workspace/data/axi/masks/0/000503090301_20230418_00020_0000001319.png', help='Path to second image')
    parser.add_argument('--output_dir', type=str, default='flow_warp_results', help='Output directory')
    parser.add_argument('--size', type=int, default=256, help='Image size (width=height)')
    # 화살표/시각화 옵션
    parser.add_argument('--arrow_stride', type=int, default=8, help='Quiver grid stride (smaller = more arrows)')
    parser.add_argument('--arrow_min_mag', type=float, default=0.0, help='Minimum magnitude threshold to draw arrows')
    parser.add_argument('--arrow_scale', type=float, default=1.0, help='Arrow scale factor (larger = longer arrows)')
    parser.add_argument('--arrow_no_norm', action='store_true', help='Do not normalize arrow vectors by magnitude')
    parser.add_argument('--save_each_method', action='store_true', help='Save warped images for each flow method')
    parser.add_argument('--swap_xy', action='store_true', help='Swap flow x/y when visualizing (diagnosed for TRUS)')
    parser.add_argument('--invert_y', action='store_true', help='Invert vertical flow for visualization')
    
    args = parser.parse_args()
    
    # 이미지 존재 확인
    if not os.path.exists(args.img1):
        print(f"Error: Image 1 not found: {args.img1}")
        return
    
    if not os.path.exists(args.img2):
        print(f"Error: Image 2 not found: {args.img2}")
        return
    
    # Optical flow와 warping 계산
    results = compute_flow_and_warp(
        args.img1,
        args.img2,
        args.output_dir,
        arrow_stride=args.arrow_stride,
        arrow_min_mag=args.arrow_min_mag,
        arrow_scale=args.arrow_scale,
        arrow_normalize=(not args.arrow_no_norm),
        save_each_method=args.save_each_method,
        swap_xy=args.swap_xy,
        invert_y=args.invert_y,
    )
    
    print(f"\nComputation completed successfully!")

if __name__ == "__main__":
    main()
