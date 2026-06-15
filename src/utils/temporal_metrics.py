import torch
import torch.nn.functional as F
import numpy as np
from .util import warp_with_flow, warp_mask_with_flow


def temporal_consistency_mse(pred_sequence, flow_sequence=None, binarize_warp: bool = True):
    """
    1 - MSE 기반 시간적 일관성 지표 계산
    - pred_sequence: List[Tensor (B,1,H,W)]
    - flow_sequence: Optional[List[Tensor (B,2,H,W)]] of len = len(pred_sequence)-1
    """
    consistency_scores = []
    for i in range(1, len(pred_sequence)):
        current_pred = pred_sequence[i]
        previous_pred = pred_sequence[i - 1]
        if flow_sequence is not None and (i - 1) < len(flow_sequence):
            flow = flow_sequence[i - 1]
            if binarize_warp:
                warped_prev = warp_mask_with_flow(previous_pred, flow)
            else:
                warped_prev = warp_with_flow(previous_pred, flow)
            consistency = 1 - F.mse_loss(current_pred, warped_prev)
        else:
            consistency = 1 - F.mse_loss(current_pred, previous_pred)
        consistency_scores.append(consistency.item())
    return float(np.mean(consistency_scores)) if consistency_scores else 0.0



