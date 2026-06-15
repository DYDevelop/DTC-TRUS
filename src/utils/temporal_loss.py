import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from src.utils.util import compute_ultrasound_optical_flow, warp_image_with_flow

class TemporalConsistencyLoss(nn.Module):
    """
    ETC-Real-time의 시간적 일관성 손실 함수
    연속된 프레임 간의 세그멘테이션 결과 일관성을 유지
    """
    
    def __init__(self, weight=0.1, temperature=1.0):
        super(TemporalConsistencyLoss, self).__init__()
        self.weight = weight
        self.temperature = temperature
        
    def forward(self, current_pred, previous_pred, flow=None):
        """
        Args:
            current_pred: 현재 프레임 예측 (B, C, H, W)
            previous_pred: 이전 프레임 예측 (B, C, H, W)
            flow: 광학 흐름 (B, 2, H, W) - 선택사항
        """
        if flow is not None:
            # 광학 흐름을 사용한 워핑
            warped_prev = self.warp_with_flow(previous_pred, flow)
            loss = F.mse_loss(current_pred, warped_prev)
        else:
            # 단순 MSE 손실
            loss = F.mse_loss(current_pred, previous_pred)
            
        return self.weight * loss
    
    def warp_with_flow(self, image, flow):
        """
        광학 흐름을 사용하여 이미지 워핑
        """
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

class TemporalKnowledgeDistillationLoss(nn.Module):
    """
    시간적 지식 증류 손실 함수
    Teacher 모델의 시간적 정보를 Student 모델에 전달
    """
    
    def __init__(self, temperature=4.0, alpha=0.7):
        super(TemporalKnowledgeDistillationLoss, self).__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')
        
    def forward(self, student_pred, teacher_pred, labels):
        """
        Args:
            student_pred: Student 모델 예측
            teacher_pred: Teacher 모델 예측
            labels: 실제 라벨
        """
        # Teacher 예측을 확률로 변환
        teacher_prob = F.softmax(teacher_pred / self.temperature, dim=1)
        
        # Student 예측을 로그 확률로 변환
        student_log_prob = F.log_softmax(student_pred / self.temperature, dim=1)
        
        # KL 손실 계산
        kl_loss = self.kl_loss(student_log_prob, teacher_prob) * (self.temperature ** 2)
        
        # 실제 라벨과의 손실
        ce_loss = F.cross_entropy(student_pred, labels)
        
        # 가중 평균
        total_loss = self.alpha * ce_loss + (1 - self.alpha) * kl_loss
        
        return total_loss

class MotionLoss(nn.Module):
    """
    모션 기반 손실 함수
    연속된 프레임 간의 모션 정보를 활용
    """
    
    def __init__(self, weight=0.1):
        super(MotionLoss, self).__init__()
        self.weight = weight
        
    def forward(self, pred1, pred2, motion_mask=None):
        """
        Args:
            pred1: 첫 번째 프레임 예측
            pred2: 두 번째 프레임 예측
            motion_mask: 모션 마스크 (선택사항)
        """
        # 예측 간 차이 계산
        diff = torch.abs(pred1 - pred2)
        
        if motion_mask is not None:
            # 모션 마스크가 있는 영역에만 손실 적용
            loss = torch.mean(diff * motion_mask)
        else:
            # 전체 영역에 손실 적용
            loss = torch.mean(diff)
            
        return self.weight * loss

def compute_temporal_consistency_metric(pred_sequence, flow_sequence=None):
    """
    시간적 일관성 메트릭 계산
    """
    consistency_scores = []
    
    for i in range(1, len(pred_sequence)):
        current_pred = pred_sequence[i]
        previous_pred = pred_sequence[i-1]
        
        if flow_sequence is not None:
            # 광학 흐름을 사용한 일관성 계산
            flow = flow_sequence[i-1]
            warped_prev = TemporalConsistencyLoss().warp_with_flow(previous_pred, flow)
            consistency = 1 - F.mse_loss(current_pred, warped_prev)
        else:
            # 단순 일관성 계산
            consistency = 1 - F.mse_loss(current_pred, previous_pred)
            
        consistency_scores.append(consistency.item())
    
    return np.mean(consistency_scores)


class UltrasoundFlowConsistency(nn.Module):
    """
    초음파 이미지에 특화된 Optical Flow 기반 시간적 일관성 손실 클래스
    """
    
    def __init__(self, method='farneback', method_kwargs=None, align_corners=True, 
                 alpha=1.0, padding_mode='border', prev_to_cur=True, 
                 sign_correction=(1.0, 1.0), ensure_dense=True, 
                 use_ultrasound_preprocess=True):
        """
        Args:
            method: optical flow 계산 방법 ('farneback', 'lucas_kanade', 'horn_schunck', 'ncc')
            method_kwargs: flow 계산 방법에 대한 추가 파라미터
            align_corners: grid_sample의 align_corners 파라미터
            alpha: 일관성 손실 가중치
            padding_mode: grid_sample의 padding_mode
            prev_to_cur: True이면 이전 프레임을 현재로 워핑, False이면 현재를 이전으로
            sign_correction: (u_sign, v_sign) 플로우 방향 보정
            ensure_dense: dense flow 보장 여부
            use_ultrasound_preprocess: 초음파 전처리 사용 여부
        """
        super(UltrasoundFlowConsistency, self).__init__()
        self.method = method
        self.method_kwargs = method_kwargs or {}
        self.align_corners = align_corners
        self.alpha = alpha
        self.padding_mode = padding_mode
        self.prev_to_cur = prev_to_cur
        self.sign_correction = sign_correction
        self.ensure_dense = ensure_dense
        self.use_ultrasound_preprocess = use_ultrasound_preprocess
        
    def compute_flow(self, img1, img2, source="image"):
        """
        Optical flow 계산
        
        Args:
            img1: 첫 번째 이미지/마스크 (B, C, H, W)
            img2: 두 번째 이미지/마스크 (B, C, H, W)
            source: "image" 또는 "mask" (현재는 동일하게 처리)
            
        Returns:
            flow: (B, 2, H, W) optical flow
        """
        kwargs = self.method_kwargs.copy()
        if self.method == 'farneback':
            # Farneback을 위한 kwargs 처리
            if 'winsize' in kwargs:
                # Farneback은 winsize를 직접 사용하지 않으므로 제거
                pass
        
        flow = compute_ultrasound_optical_flow(
            img1, img2, 
            method=self.method, 
            method_kwargs=kwargs if kwargs else None
        )
        
        # Sign correction
        if self.sign_correction != (1.0, 1.0):
            u_sign, v_sign = self.sign_correction
            flow = flow * torch.tensor([u_sign, v_sign], device=flow.device).view(1, 2, 1, 1)
        
        return flow
    
    def consistency_loss(self, current_pred, previous_pred, current_img=None, previous_img=None,
                        flow_source_pred="mask", flow_source_img="image",
                        flow_prev2cur_pred=None, flow_prev2cur_img=None,
                        reduction="mean"):
        """
        시간적 일관성 손실 계산
        
        Args:
            current_pred: 현재 프레임 예측 (B, C, H, W)
            previous_pred: 이전 프레임 예측 (B, C, H, W)
            current_img: 현재 프레임 이미지 (B, C, H, W) - NOC 계산용
            previous_img: 이전 프레임 이미지 (B, C, H, W) - NOC 계산용
            flow_source_pred: 예측에 사용할 flow 소스 ("mask" 또는 "image")
            flow_source_img: 이미지에 사용할 flow 소스 ("mask" 또는 "image")
            flow_prev2cur_pred: 예측용 flow (None이면 자동 계산)
            flow_prev2cur_img: 이미지용 flow (None이면 자동 계산)
            reduction: "mean" 또는 "sum"
            
        Returns:
            loss: 일관성 손실
        """
        # Flow 계산
        if flow_prev2cur_pred is None:
            if flow_source_pred == "mask":
                flow_prev2cur_pred = self.compute_flow(previous_pred, current_pred, source="mask")
            else:
                flow_prev2cur_pred = self.compute_flow(previous_pred, current_pred, source="image")
        
        if flow_prev2cur_img is None and current_img is not None:
            if flow_source_img == "mask":
                flow_prev2cur_img = self.compute_flow(previous_img, current_img, source="mask")
            else:
                flow_prev2cur_img = self.compute_flow(previous_img, current_img, source="image")
        
        # 예측 워핑
        if self.prev_to_cur:
            warped_prev_pred = self._warp(previous_pred, flow_prev2cur_pred)
        else:
            # 현재를 이전으로 워핑 (역방향)
            flow_cur2prev_pred = -flow_prev2cur_pred
            warped_prev_pred = self._warp(current_pred, flow_cur2prev_pred)
            current_pred = previous_pred  # 비교 대상 변경
        
        # NOC (Non-Occlusion) 마스크 계산 (이미지가 제공된 경우)
        if current_img is not None and previous_img is not None:
            if self.prev_to_cur:
                warped_prev_img = self._warp(previous_img, flow_prev2cur_img)
            else:
                flow_cur2prev_img = -flow_prev2cur_img
                warped_prev_img = self._warp(current_img, flow_cur2prev_img)
                current_img = previous_img
            
            # 이미지 차이 계산 (NOC)
            img_diff = torch.abs(torch.sum(current_img - warped_prev_img, dim=1, keepdim=True))
            noc_mask = torch.exp(-self.alpha * img_diff)
        else:
            noc_mask = torch.ones_like(current_pred[:, :1, :, :])
        
        # 예측 차이 계산
        pred_diff = torch.abs(current_pred - warped_prev_pred)
        
        # NOC 가중 일관성 손실
        consistency_loss = 1.0 - torch.mean(noc_mask * (1.0 - pred_diff))
        
        return consistency_loss
    
    def _warp(self, image, flow):
        """
        이미지/마스크를 flow로 워핑
        
        Args:
            image: (B, C, H, W) 이미지 또는 마스크
            flow: (B, 2, H, W) optical flow
            
        Returns:
            warped: (B, C, H, W) 워핑된 이미지/마스크
        """
        B, C, H, W = image.shape
        
        # Flow 크기 조정 (필요한 경우)
        if flow.shape[-2:] != (H, W):
            if flow.shape[-2:] == (1, 1):
                flow = flow.expand(B, 2, H, W)
            else:
                flow = F.interpolate(flow, size=(H, W), mode='bilinear', align_corners=self.align_corners)
        
        # 정규화 그리드 생성 [-1, 1]
        ys = torch.linspace(-1.0, 1.0, H, device=image.device, dtype=image.dtype)
        xs = torch.linspace(-1.0, 1.0, W, device=image.device, dtype=image.dtype)
        base_y, base_x = torch.meshgrid(ys, xs, indexing='ij') if hasattr(torch, 'meshgrid') else torch.meshgrid(ys, xs)
        base_grid = torch.stack((base_x, base_y), dim=-1)  # (H, W, 2)
        base_grid = base_grid.unsqueeze(0).repeat(B, 1, 1, 1)  # (B, H, W, 2)
        
        # 픽셀 flow를 정규화 델타로 변환
        dx = flow[:, 0, :, :]
        dy = flow[:, 1, :, :]
        dx_norm = (2.0 * dx) / max(W - 1, 1)
        dy_norm = (2.0 * dy) / max(H - 1, 1)
        flow_norm = torch.stack((dx_norm, dy_norm), dim=-1)  # (B, H, W, 2)
        
        grid = base_grid + flow_norm
        warped = F.grid_sample(
            image, grid, 
            mode='bilinear', 
            padding_mode=self.padding_mode, 
            align_corners=self.align_corners
        )
        
        return warped 