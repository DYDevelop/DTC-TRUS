import os
import argparse
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torch.multiprocessing as mp
from tqdm import tqdm
from src.load_model import get_model
from src.utils.util import *
from src.dataloader.temporal_dataset_ddp import getDataloader
import src.utils.losses as losses
from src.utils.util import AverageMeter 
from src.utils.metrics import iou_score
from src.utils.metrics_SUN import evaluator as evaluator_SUN
import albumentations as A
from PIL import Image

import warnings
warnings.filterwarnings('ignore')
torch.set_float32_matmul_precision('high')

def seed_torch(seed):
    """재현성을 위한 시드 설정"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def knowledge_distillation_loss(student_logits, teacher_logits, temperature=4.0):
    """
    Knowledge Distillation Loss for Binary Segmentation
    Temperature-scaled MSE on soft predictions
    """
    student_probs = torch.sigmoid(student_logits / temperature)
    teacher_probs = torch.sigmoid(teacher_logits / temperature)
    kd_loss = torch.nn.functional.mse_loss(student_probs, teacher_probs)
    return kd_loss * (temperature ** 2)


def get_teacher_model(args, device):
    """사전 학습된 모델을 teacher로 사용"""
    teacher = get_model(args)
    
    # 체크포인트 로드 (필수!)
    if args.checkpoint and os.path.exists(args.checkpoint):
        state_dict = torch.load(args.checkpoint, map_location=device)
        # DDP 체크포인트인 경우 'module.' prefix 제거
        if list(state_dict.keys())[0].startswith('module.'):
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        teacher.load_state_dict(state_dict)
        print(f"✓ Teacher model loaded from: {args.checkpoint}")
    else:
        raise FileNotFoundError(f"❌ Teacher checkpoint not found: {args.checkpoint}")
    
    teacher.to(device)
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad = False
    
    return teacher


@torch.no_grad()
def generate_flip_pseudo_label(model, images):
    """
    Flip-based Pseudo Label 생성 (최적화)
    원본 이미지와 좌우 반전된 이미지를 모두 모델에 입력하여 더 정확한 pseudo label 생성
    """
    was_training = model.training
    model.eval()
    
    batch_size, sequence_length = images.shape[:2]
    device = images.device
    
    # 모든 시퀀스를 한번에 처리 (메모리 효율적)
    all_images = images.view(-1, *images.shape[2:])  # (B*T, C, H, W)
    
    # 원본 예측
    outputs_original, _, _ = model(all_images)
    masks_original = torch.sigmoid(outputs_original)
    
    # Flip 예측
    images_flipped = torch.flip(all_images, dims=[3])
    outputs_flipped, _, _ = model(images_flipped)
    masks_flipped = torch.sigmoid(outputs_flipped)
    masks_unflipped = torch.flip(masks_flipped, dims=[3])
    
    # 평균을 통한 안정적인 pseudo label (soft probability)
    pseudo_masks = (masks_original + masks_unflipped) / 2.0
    
    # 원래 shape으로 복원
    pseudo_masks = pseudo_masks.view(batch_size, sequence_length, *pseudo_masks.shape[1:])
    
    if was_training:
        model.train()
    
    return pseudo_masks

def random_tta(prob=0.5): # KD 할때 적용하는 것도 고려
    return A.Compose([
        A.HorizontalFlip(p=prob),
        A.VerticalFlip(p=prob),
        A.RandomBrightnessContrast(p=prob),
        A.GaussianBlur(p=prob),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1,
                        rotate_limit=10, p=prob),
        A.Cutout(num_holes=4, max_h_size=16, max_w_size=16, p=prob),
    ])

def masked_avg_pool(feat, mask, eps=1e-6):
    # feat: (B,C,H,W), mask: (B,1,H,W)
    masked = feat * mask
    denom = mask.sum(dim=[2,3], keepdim=True).clamp(min=eps)
    pooled = masked.sum(dim=[2,3], keepdim=True) / denom  # (B,C,1,1)
    return pooled.squeeze(-1).squeeze(-1)                 # (B,C)

def get_adaptive_threshold(mask_probs, percentile=0.7):
    """
    Batch별로 adaptive threshold 계산
    
    Args:
        mask_probs: (B, 1, H, W)
        percentile: 0.7 = top 30%
    
    Returns:
        thresholds: (B,) scalar per batch
    """
    B = mask_probs.shape[0]
    thresholds = []
    
    for b in range(B):
        sorted_probs = torch.sort(mask_probs[b].flatten())[0]
        idx = int(len(sorted_probs) * percentile)
        thresholds.append(sorted_probs[idx])
    
    return torch.stack(thresholds).mean().item()  # Batch 평균

def train_epoch(trainloader, valloader, model, teacher_model, optimizer, criterion, 
                epoch, args, scaler=None, rank=0):
    """
    수정된 훈련 에포크 - Pseudo Label 문제 해결
    """
    model.train()
    teacher_model.eval()
    
    avg_meters = {
        'total_loss': AverageMeter(),
        'seg_loss': AverageMeter(),
        'kd_loss': AverageMeter(),
        'temporal_loss': AverageMeter(),
        'proto_loss' : AverageMeter(),
        'consistency': AverageMeter(),
        'val_loss': AverageMeter(),
        'val_iou': AverageMeter(),
        'val_dsc': AverageMeter(),
        'val_SE': AverageMeter(),
        'val_PC': AverageMeter(),
        'val_F1': AverageMeter(),
        'val_ACC': AverageMeter(),
        'val_temporal_consistency': AverageMeter(),

        'val_Sm': AverageMeter(),
        'val_mEm': AverageMeter(),
        'val_wF': AverageMeter(),
        'val_mFm': AverageMeter(),
        'val_mD': AverageMeter(),
        'val_mS': AverageMeter(),
    }
    
    use_amp = scaler is not None
    
    if rank == 0:
        pbar = tqdm(trainloader, desc=f"Epoch {epoch}/{args.epoch} [Train]")
    else:
        pbar = trainloader
    
    for i_batch, sampled_batch in enumerate(pbar):
        images = sampled_batch['images']  # (B, T, C, H, W)
        batch_size, sequence_length = images.shape[:2]
        
        images = images.cuda(non_blocking=True)
        
        if i_batch % args.gradient_accumulation_steps == 0:
            optimizer.zero_grad()
        
        with torch.cuda.amp.autocast(enabled=use_amp):
            all_images = images.view(-1, *images.shape[2:])  # (B*T, C, H, W)
            
            # ===== 수정 1: Student로  Augmented-Consistency 추가 =====
            model.eval()
            with torch.no_grad():
                # Flip 예측
                images_flipped = torch.flip(all_images, dims=[3])
                outputs_flipped, _, _ = model(images_flipped)
                masks_flipped = torch.sigmoid(outputs_flipped)
                masks_unflipped = torch.flip(masks_flipped, dims=[3])
                
                # # Hard label로 변환 (threshold=0.5)
                # augmented_labels_hard = (masks_unflipped > 0.5).float()
            
            model.train()
            # ===== 수정 2: Student Forward Pass =====
            student_outputs, student_local_features, student_global_features = model(all_images)  # (B*T, 1, H, W) logits
            sequence_outputs = student_outputs.view(batch_size, sequence_length, *student_outputs.shape[1:])
            seq_stu_loc = student_local_features.view(batch_size, sequence_length, *student_local_features.shape[1:])
            seq_stu_glob = student_global_features.view(batch_size, sequence_length, *student_global_features.shape[1:])
            
            # ===== 3: Segmentation Loss - Hard Label  =====
            # seg_loss = criterion(student_outputs, augmented_labels_hard)

            # ===== 3: Segmentation Loss - Soft Label  =====
            seg_loss = criterion(student_outputs, masks_unflipped.detach())
            
            # ===== 4: Knowledge Distillation Loss =====
            # Teacher의 soft probability와 student logit 비교
            with torch.no_grad():
                teacher_outputs, _, _ = teacher_model(all_images)
            kd_loss = knowledge_distillation_loss(student_outputs, teacher_outputs, args.temperature)
            
            # ===== 5. Temporal Consistency Loss =====
            temporal_loss, L_proto_temp = 0.0, 0.0
            consistency_sum = 0.0
            if sequence_length > 1:
                sequence_probs = torch.sigmoid(sequence_outputs)  # (B, T, 1, H, W)
                
                for t in range(1, sequence_length):
                    prev_img = images[:, t-1]
                    curr_img = images[:, t]
                    prev_mask = sequence_probs[:, t-1]
                    curr_mask = sequence_probs[:, t]
                    
                    # Optical Flow 계산
                    flow = compute_optical_flow(prev_img, curr_img, modality='rgb', method=args.flow_method)
                    
                    # Warping
                    warp_prev_img = warp_image_with_flow(prev_img, flow)
                    warp_prev_mask = warp_mask_with_flow(prev_mask, flow)
                    
                    # NOC (Non-Occlusion) 마스크
                    img_diff = torch.abs(torch.sum(curr_img - warp_prev_img, dim=1, keepdim=True))
                    noc_mask = torch.exp(-1.0 * img_diff) # 이미지 차이에 기반한 가중치 맵 (크면 작아짐)
                    
                    # Consistency Loss
                    mask_diff = torch.abs(curr_mask - warp_prev_mask)
                    t_loss = 1.0 - torch.mean(noc_mask * (1.0 - mask_diff))
                    temporal_loss += t_loss
                    consistency_sum += (1.0 - t_loss.item())

                    """
                      Using Local & Global Prototype Feature for contrastive learning
                      - Local : Last hidden feature map before seg head
                      - Global : Bottleneck feature map
                      - Cosine-similarity loss
                    """
                    # -----  Prototype Feature map Contrastive Learning -----
                    prev_local_feat = seq_stu_loc[:, t-1]
                    curr_local_feat = seq_stu_loc[:, t]
                    prev_global_feat = seq_stu_glob[:, t-1]
                    curr_global_feat = seq_stu_glob[:, t]

                    thr = get_adaptive_threshold(curr_mask) # Using adaptive threshold
                    prev_fg = (prev_mask > thr).float()   # (B,1,H,W)
                    curr_fg = (curr_mask > thr).float()

                    prev_bg = 1.0 - prev_fg
                    curr_bg = 1.0 - curr_fg

                    # ------ Local Prototype Contrastive ------ --> Using optical flow to map local faetures
                    prev_local_feat_warp = warp_feature_with_flow(prev_local_feat, flow)
                    warp_prev_fg = (warp_prev_mask > thr).float()
                    fg_mask_for_proto = (curr_fg * warp_prev_fg) # Using intersection of two mask (smoother one)

                    proto_prev_local = masked_avg_pool(prev_local_feat_warp, fg_mask_for_proto)   # (B,C_loc)
                    proto_curr_local = masked_avg_pool(curr_local_feat, fg_mask_for_proto)   # (B,C_loc)3

                    sim_local = F.cosine_similarity(proto_prev_local, proto_curr_local, dim=-1)  # (B,)
                    L_local_temp = (1.0 - sim_local).mean()

                    # ------ Global Prototype Contrastive ------ --> Scene-level 전체에 대한 BG 통계가 비슷하게 학습 (optical flow X)
                    prev_bg_b = F.interpolate(prev_bg, size=prev_global_feat.shape[-2:], mode='nearest')
                    curr_bg_b = F.interpolate(curr_bg, size=curr_global_feat.shape[-2:], mode='nearest')

                    proto_prev_glob = masked_avg_pool(prev_global_feat, prev_bg_b)  # (B,C_glob)
                    proto_curr_glob = masked_avg_pool(curr_global_feat, curr_bg_b)  # (B,C_glob)

                    sim_glob = F.cosine_similarity(proto_prev_glob, proto_curr_glob, dim=-1)
                    L_global_temp = (1.0 - sim_glob).mean()

                    # ------ Local and Global Weighted Sum ------
                    fg_frac_t = curr_mask[:,0].mean(dim=[1,2]) # (B,)
                    bg_frac_t = 1.0 - fg_frac_t
                    w_fg_t = fg_frac_t.mean().detach()
                    w_bg_t = bg_frac_t.mean().detach()
                    L_proto_temp += w_bg_t * L_local_temp + w_fg_t * L_global_temp # 비율에 알맞게 weighted sum

                temporal_loss = temporal_loss / (sequence_length - 1)
                avg_consistency = consistency_sum / (sequence_length - 1)
                L_proto_temp = L_proto_temp / (sequence_length - 1)
            else:
                avg_consistency = 0.0
            
            # Total Loss
            total_loss = (seg_loss * args.seg_lam + 
                         kd_loss * args.kd_lam + 
                         temporal_loss * args.con_lam +
                         L_proto_temp * args.proto_lam)
            
            total_loss = total_loss / args.gradient_accumulation_steps
        
        # Backward pass
        if use_amp:
            scaler.scale(total_loss).backward()
        else:
            total_loss.backward()
        
        # Optimizer step
        if (i_batch + 1) % args.gradient_accumulation_steps == 0:
            if use_amp:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
        
        # 메트릭 업데이트
        avg_meters['total_loss'].update(total_loss.item() * args.gradient_accumulation_steps, batch_size)
        avg_meters['seg_loss'].update(seg_loss.item(), batch_size)
        avg_meters['kd_loss'].update(kd_loss.item(), batch_size)
        avg_meters['temporal_loss'].update(temporal_loss.item() if sequence_length > 1 else 0.0, batch_size)
        avg_meters['proto_loss'].update(L_proto_temp.item(), batch_size)
        avg_meters['consistency'].update(avg_consistency, batch_size)
        
        # Progress bar 업데이트
        if rank == 0 and i_batch % 10 == 0:
            pbar.set_postfix({
                'loss': f"{avg_meters['total_loss'].avg:.4f}",
                'seg': f"{avg_meters['seg_loss'].avg:.4f}",
                'kd': f"{avg_meters['kd_loss'].avg:.4f}",
                'proto': f"{avg_meters['proto_loss'].avg:.4f}",
                'temp': f"{avg_meters['temporal_loss'].avg:.4f}"
            })
    
    # Validation (기존과 동일)
    model.eval()
    if rank == 0:
        val_pbar = tqdm(valloader, desc=f"Epoch {epoch}/{args.epoch} [Val]")
    else:
        val_pbar = valloader
    
    with torch.no_grad():
        for sampled_batch in val_pbar:
            img_batchs = sampled_batch['images'].cuda(non_blocking=True)
            label_batchs = sampled_batch['labels'].cuda(non_blocking=True)
            img_path_batchs = sampled_batch['image_paths']
            batch_size, sequence_length = img_batchs.shape[:2]
            
            sequence_masks = []
            
            for seq_idx in range(sequence_length):
                img_batch = img_batchs[:, seq_idx]
                label_batch = label_batchs[:, seq_idx]
                
                with torch.cuda.amp.autocast(enabled=use_amp):
                    output, _, _ = model(img_batch)
                    loss = criterion(output, label_batch)
                
                pred_mask = torch.sigmoid(output)
                sequence_masks.append(pred_mask)
                
                iou, dice, SE, PC, F1, _, ACC = iou_score(output, label_batch)
                sun_results = evaluator_SUN(label_batch, output, metrics=["Smeasure", "meanEm", "wFmeasure", "meanFm", "meanDice", "meanSen"])
                
                for b_idx in range(img_batch.size(0)):
                    res_numpy = pred_mask[b_idx, ...][0].squeeze().cpu().numpy()
                    save_name = './res/TRUS_V/'+ img_path_batchs[seq_idx][b_idx].split('/')[-1]
                    os.makedirs(os.path.dirname(save_name), exist_ok=True)
                    # 이미지 저장 (0~1 실수값이면 255 곱하기)
                    save_img = (res_numpy * 255).astype(np.uint8)
                    Image.fromarray(save_img).save(save_name)
                
                avg_meters['val_loss'].update(loss.item(), img_batch.size(0))
                avg_meters['val_iou'].update(iou, img_batch.size(0))
                avg_meters['val_dsc'].update(dice, img_batch.size(0))
                avg_meters['val_SE'].update(SE, img_batch.size(0))
                avg_meters['val_PC'].update(PC, img_batch.size(0))
                avg_meters['val_F1'].update(F1, img_batch.size(0))
                avg_meters['val_ACC'].update(ACC, img_batch.size(0))

                avg_meters['val_Sm'].update(sun_results['Smeasure'], img_batch.size(0))
                avg_meters['val_mEm'].update(sun_results['meanEm'], img_batch.size(0))
                avg_meters['val_wF'].update(sun_results['wFmeasure'], img_batch.size(0))
                avg_meters['val_mFm'].update(sun_results['meanFm'], img_batch.size(0))
                avg_meters['val_mD'].update(sun_results['meanDice'], img_batch.size(0))
                avg_meters['val_mS'].update(sun_results['meanSen'], img_batch.size(0))
            
            # Temporal Consistency 평가
            if sequence_length > 1:
                temporal_consistency = 0.0
                for t in range(1, sequence_length):
                    mask_diff = torch.abs(sequence_masks[t] - sequence_masks[t-1])
                    temporal_consistency += (1.0 - mask_diff.mean().item())
                avg_meters['val_temporal_consistency'].update(
                    temporal_consistency / (sequence_length - 1), batch_size
                )
    
    return avg_meters


def setup_ddp(rank, world_size):
    """DDP 초기화"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12345'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup_ddp():
    """DDP 종료"""
    dist.destroy_process_group()


def train_worker(rank, world_size, args):
    """Multi-GPU 학습 워커"""
    # DDP 설정
    if args.multi_gpu:
        setup_ddp(rank, world_size)
    
    # 시드 설정 (각 프로세스마다 다른 시드)
    seed_torch(args.seed + rank)
    
    device = torch.device(f'cuda:{rank}')
    
    # DDP 정보를 args에 추가
    args.rank = rank
    args.world_size = world_size if args.multi_gpu else 1
    
    # 데이터 로더
    trainloader, valloader, testloader = getDataloader(args)
    
    # Student 모델
    model = get_model(args).to(device)
    
    # Multi-GPU 설정
    if args.multi_gpu:
        model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=False)
        model_without_ddp = model.module
    else:
        model_without_ddp = model
    
    # Teacher 모델
    teacher_model = get_teacher_model(args, device)
    
    # Optimizer & Scheduler
    optimizer = optim.AdamW(
        params=filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.base_lr,
        weight_decay=1e-4
    )
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )
    
    # Loss function
    criterion = losses.__dict__['BCEDiceLoss']().to(device)
    
    # Mixed Precision Scaler
    scaler = torch.cuda.amp.GradScaler() if args.use_amp else None
    
    if rank == 0:
        print(f"{'='*60}")
        print(f"Training Configuration:")
        print(f"  Model: {args.model}")
        print(f"  GPUs: {world_size if args.multi_gpu else 1}")
        print(f"  Batch size per GPU: {args.batch_size}")
        print(f"  Total batch size: {args.batch_size * (world_size if args.multi_gpu else 1)}")
        print(f"  Mixed Precision: {args.use_amp}")
        print(f"  Gradient Accumulation: {args.gradient_accumulation_steps}")
        print(f"  Iterations per epoch: {len(trainloader)}")
        print(f"{'='*60}\n")
    
    # 학습 시작
    best_dsc = float("-inf")
    patience = 0
    os.makedirs(args.ckpt_root, exist_ok=True)
    
    for epoch_num in range(args.epoch):
        # DDP: 에포크마다 sampler 셔플
        if args.multi_gpu:
            trainloader.sampler.set_epoch(epoch_num)
        
        # 학습
        train_meters = train_epoch(
            trainloader, valloader, model, teacher_model, 
            optimizer, criterion, epoch_num, args, scaler, rank
        )
        
        # Learning rate update
        scheduler.step()
        
        # Rank 0에서만 로깅 및 체크포인트 저장
        if rank == 0:
            print(f'\n{"="*60}')
            print(f'Epoch [{epoch_num+1}/{args.epoch}] Summary:')
            print(f'  Train:')
            print(f'    Total Loss: {train_meters["total_loss"].avg:.4f}')
            print(f'    Seg Loss: {train_meters["seg_loss"].avg:.4f}')
            print(f'    KD Loss: {train_meters["kd_loss"].avg:.4f}')
            print(f'    Temporal Loss: {train_meters["temporal_loss"].avg:.4f}')
            print(f'    Consistency: {train_meters["consistency"].avg:.4f}')
            print(f'  Val:')
            print(f'    Loss: {train_meters["val_loss"].avg:.4f}')
            print(f'    IoU: {train_meters["val_iou"].avg:.4f}')
            print(f'    DSC: {train_meters["val_dsc"].avg:.4f}')
            print(f'    SE: {train_meters["val_SE"].avg:.4f}')
            print(f'    PC: {train_meters["val_PC"].avg:.4f}')
            print(f'    F1: {train_meters["val_F1"].avg:.4f}')
            print(f'    ACC: {train_meters["val_ACC"].avg:.4f}')
            print(f'    Temporal Consistency: {train_meters["val_temporal_consistency"].avg:.4f}')
            print(f'{"="*60}')
            print(f'  SUN Val:')
            print(f'    S-measure: {train_meters["val_Sm"].avg:.4f}')
            print(f'    mean E-measure: {train_meters["val_mEm"].avg:.4f}')
            print(f'    weighted F-measure: {train_meters["val_wF"].avg:.4f}')
            print(f'    mean F-measure: {train_meters["val_mFm"].avg:.4f}')
            print(f'    mean Dice: {train_meters["val_mD"].avg:.4f}')
            print(f'    mean Sensitivity: {train_meters["val_mS"].avg:.4f}')
            print(f'{"="*60}')

            # Best model 저장
            if train_meters["val_dsc"].avg > best_dsc:
                best_dsc = train_meters["val_dsc"].avg
                patience = 0
                ckpt_path = os.path.join(args.ckpt_root, f"{args.model}_best.pth")
                torch.save({
                    'epoch': epoch_num,
                    'model_state_dict': model_without_ddp.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_dsc': best_dsc,
                    'args': args
                }, ckpt_path)
                print(f"✓ Best model saved: DSC = {best_dsc:.4f}")
            else:
                patience += 1
            
            # Early stopping
            if patience >= args.patience:
                print(f"⚠ Early stopping triggered (patience={patience})")
                break
    
    # 최종 체크포인트 저장
    if rank == 0:
        final_ckpt_path = os.path.join(args.ckpt_root, f"{args.model}_final.pth")
        torch.save(model_without_ddp.state_dict(), final_ckpt_path)
        
        # Lambda 설정 저장
        lambda_map = {
            "seg_lam": args.seg_lam,
            "kd_lam": args.kd_lam,
            "con_lam": args.con_lam,
            "proto_lam": args.proto_lam,
            "temp": args.temperature,
            "best_dsc": best_dsc,

            "S-measure": train_meters["val_Sm"].avg,
            "mean Dice": train_meters["val_mD"].avg,
        }
        with open(os.path.join(args.ckpt_root, "training_config.json"), "w") as f:
            json.dump(lambda_map, f, indent=2)
        
        print(f"\n{'='*60}")
        print(f"Training Finished!")
        print(f"  Best DSC: {best_dsc:.4f}")
        print(f"  Final model: {final_ckpt_path}")
        print(f"{'='*60}")
    
    # DDP 종료
    if args.multi_gpu:
        cleanup_ddp()


def main(args):
    """메인 함수"""
    # GPU 개수 확인
    if args.multi_gpu:
        world_size = torch.cuda.device_count()
        if world_size < 2:
            print("⚠ Multi-GPU mode requested but only 1 GPU available. Using single GPU.")
            args.multi_gpu = False
            world_size = 1
        else:
            print(f"✓ Using {world_size} GPUs for training")
    else:
        world_size = 1
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    
    # Multi-GPU training
    if args.multi_gpu and world_size > 1:
        mp.spawn(train_worker, args=(world_size, args), nprocs=world_size, join=True)
    else:
        train_worker(0, 1, args)


if __name__ == "__main__":
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
    
    parser = argparse.ArgumentParser(description='Optimized Multi-GPU Video Segmentation Training')
    
    # 모델 설정
    parser.add_argument('--model', type=str, default="U_Net", help='model architecture')
    parser.add_argument('--use_prototype', action='store_true', help='return feature maps')
    parser.add_argument('--checkpoint', type=str, 
                       default=os.path.join(PROJECT_ROOT, "image_checkpoint", "checkpoint_axi", "U_Net_0_model.pth"),
                       help='teacher model checkpoint')
    
    # 데이터 설정
    parser.add_argument('--base_dir', type=str, default=os.path.join(PROJECT_ROOT, "etc_dataset"))
    parser.add_argument('--dataset', type=str, default="SUN")
    parser.add_argument('--csv_dir', type=str, 
                       default=os.path.join(PROJECT_ROOT, "etc_dataset", "list", "etc_dataset.csv"))
    parser.add_argument('--csv_val_dir', type=str, default=None)
    parser.add_argument('--ckpt_root', type=str, 
                       default=os.path.join(SCRIPT_DIR, "checkpoint", "flip_kd_optimized"))
    
    # 학습 설정
    parser.add_argument('--base_lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size per GPU')
    parser.add_argument('--epoch', type=int, default=50, help='training epochs')
    parser.add_argument('--patience', type=int, default=10, help='early stopping patience')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    
    # 이미지 설정
    parser.add_argument('--img_size', type=int, nargs=2, default=[256, 448])
    parser.add_argument('--num_classes', type=int, default=1)
    
    # Temporal 설정
    parser.add_argument('--sequence_length', type=int, default=3)
    parser.add_argument('--frame_gap', type=int, default=1)
    parser.add_argument('--temporal_weight', type=float, default=0.1)
    parser.add_argument('--flow_method', type=str, default="farneback",
                       choices=['lucas_kanade', 'horn_schunck', 'farneback', 'ncc'])
    
    # Loss 가중치 (최적화된 기본값)
    parser.add_argument('--seg_lam', type=float, default=3.0, help='segmentation loss weight')
    parser.add_argument('--kd_lam', type=float, default=1.0, help='knowledge distillation loss weight')
    parser.add_argument('--con_lam', type=float, default=2.0, help='temporal consistency loss weight')
    parser.add_argument('--proto_lam', type=float, default=0.1, help='prototype contrastive loss weight')
    parser.add_argument('--temperature', type=float, default=4.0, help='KD temperature')
    
    # 최적화 설정
    parser.add_argument('--multi_gpu', action='store_true', help='use multi-GPU training (DDP)')
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID for single GPU training')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                       help='gradient accumulation steps')
    parser.add_argument('--k_fold', type=int, default=1)
    
    args = parser.parse_args()
    
    # 체크포인트 디렉토리 생성
    os.makedirs(args.ckpt_root, exist_ok=True)
    
    main(args)

"""
사용 예시:

# Single GPU (기본)
python vid_main_flip_kd_proto_ddp.py

# Single GPU with mixed precision
python vid_main_flip_kd_proto_ddp.py --use_amp

# Multi-GPU (모든 가용 GPU 사용)
python vid_main_flip_kd_proto_ddp.py --multi_gpu --use_amp

# Multi-GPU with gradient accumulation (효과적인 배치 크기 증가)
python vid_main_flip_kd_proto_ddp.py --multi_gpu --use_amp --batch_size 4 --gradient_accumulation_steps 2

# 특정 GPU만 사용 (Single GPU)
CUDA_VISIBLE_DEVICES=0 python vid_main_flip_kd_proto_ddp.py --use_amp

# 특정 GPU들만 사용 (Multi-GPU)
CUDA_VISIBLE_DEVICES=0,1,2 python vid_main_flip_kd_proto_ddp.py --multi_gpu --use_amp

"""