import numpy as np
import torch
import cv2
from scipy import ndimage


def preprocess_ultrasound_for_flow(img):
    """
    초음파 이미지를 optical flow 계산에 최적화
    입력: float32 [0,1], shape (H, W)
    출력: float32 [0,1], shape (H, W)
    """
    img_denoised = cv2.bilateralFilter((img * 255).astype(np.uint8), 9, 75, 75).astype(np.float32) / 255.0
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_enhanced = clahe.apply((img_denoised * 255).astype(np.uint8)).astype(np.float32) / 255.0
    img_smooth = cv2.GaussianBlur(img_enhanced, (3, 3), 0.5)
    return img_smooth


def median_filter_flow(flow, kernel_size=3):
    return ndimage.median_filter(flow, size=kernel_size)


def compute_ncc_block_flow(img1, img2, block_size=21, search_radius=7, stride=4):
    """
    Speckle tracking: NCC 기반 블록 매칭으로 dense optical flow 근사 계산
    입력: float32 [0,1], 단일 채널 (H, W)
    반환: (2, H, W) [u, v]
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
    Ix = cv2.Sobel(img1, cv2.CV_64F, 1, 0, ksize=3) * 0.125
    Iy = cv2.Sobel(img1, cv2.CV_64F, 0, 1, ksize=3) * 0.125
    It = img2 - img1

    w = window_size // 2
    H, W = img1.shape
    u = np.zeros((H, W))
    v = np.zeros((H, W))

    for y in range(w, H - w):
        for x in range(w, W - w):
            Ix_win = Ix[y-w:y+w+1, x-w:x+w+1].flatten()
            Iy_win = Iy[y-w:y+w+1, x-w:x+w+1].flatten()
            It_win = It[y-w:y+w+1, x-w:x+w+1].flatten()
            A = np.column_stack([Ix_win, Iy_win])
            b = -It_win
            ATA = A.T @ A
            ATb = A.T @ b
            if np.linalg.det(ATA) > 1e-6:
                ATA += np.eye(2) * 1e-6
                flow_vec = np.linalg.solve(ATA, ATb)
                u[y, x] = flow_vec[0]
                v[y, x] = flow_vec[1]

    u = median_filter_flow(u)
    v = median_filter_flow(v)
    return np.stack([u, v], axis=0)


def compute_farneback_flow(img1, img2):
    img1_uint8 = (img1 * 255).astype(np.uint8)
    img2_uint8 = (img2 * 255).astype(np.uint8)
    flow = cv2.calcOpticalFlowFarneback(
        img1_uint8,
        img2_uint8,
        None,
        0.5,
        3,
        15,
        3,
        5,
        1.2,
        0
    )
    flow_u = median_filter_flow(flow[:, :, 0])
    flow_v = median_filter_flow(flow[:, :, 1])
    return np.stack([flow_u, flow_v], axis=0)


def compute_horn_schunck_flow(img1, img2, alpha=0.1, iterations=100):
    Ix = cv2.Sobel(img1, cv2.CV_64F, 1, 0, ksize=3) * 0.125
    Iy = cv2.Sobel(img1, cv2.CV_64F, 0, 1, ksize=3) * 0.125
    It = img2 - img1
    u = np.zeros_like(img1)
    v = np.zeros_like(img1)
    for _ in range(iterations):
        u_avg = cv2.GaussianBlur(u, (3, 3), 0)
        v_avg = cv2.GaussianBlur(v, (3, 3), 0)
        denominator = alpha**2 + Ix**2 + Iy**2 + 1e-8
        numerator = Ix * u_avg + Iy * v_avg + It
        u = u_avg - Ix * numerator / denominator
        v = v_avg - Iy * numerator / denominator
    return np.stack([u, v], axis=0)


@torch.no_grad()
def compute_ultrasound_optical_flow(img1, img2, method='farneback', method_kwargs=None):
    """
    초음파 이미지에 특화된 optical flow 계산
    입력: img1, img2 as torch.Tensor (B, C, H, W)
    반환: torch.Tensor (B, 2, H, W)
    """
    if torch.is_tensor(img1):
        img1_np = img1.squeeze(1).detach().cpu().numpy() if img1.shape[1] == 1 else img1[:, 0].detach().cpu().numpy()
        img2_np = img2.squeeze(1).detach().cpu().numpy() if img2.shape[1] == 1 else img2[:, 0].detach().cpu().numpy()
        is_tensor = True
    else:
        img1_np = img1
        img2_np = img2
        is_tensor = False

    if img1_np.ndim == 2:
        img1_np = img1_np[np.newaxis, ...]
        img2_np = img2_np[np.newaxis, ...]
    elif img1_np.ndim == 3:
        pass
    else:
        raise ValueError(f"Unsupported image dimensions for flow computation: {img1_np.shape}")

    batch_size = img1_np.shape[0]
    flows = []
    kwargs = method_kwargs or {}

    for b in range(batch_size):
        img1_proc = preprocess_ultrasound_for_flow(img1_np[b])
        img2_proc = preprocess_ultrasound_for_flow(img2_np[b])
        if method == 'lucas_kanade':
            flow = compute_lucas_kanade_flow(img1_proc, img2_proc, **{k: kwargs[k] for k in kwargs if k in ['window_size']})
        elif method == 'horn_schunck':
            flow = compute_horn_schunck_flow(img1_proc, img2_proc, **{k: kwargs[k] for k in kwargs if k in ['alpha', 'iterations']})
        elif method == 'ncc':
            flow = compute_ncc_block_flow(img1_proc, img2_proc,
                                          block_size=int(kwargs.get('block_size', 21)),
                                          search_radius=int(kwargs.get('search_radius', 7)),
                                          stride=int(kwargs.get('stride', 4)))
        else:
            flow = compute_farneback_flow(img1_proc, img2_proc)
        flows.append(flow)

    flow_array = np.stack(flows, axis=0)
    if is_tensor:
        flow_tensor = torch.from_numpy(flow_array).float()
        if img1.is_cuda:
            flow_tensor = flow_tensor.cuda()
        return flow_tensor
    return flow_array





