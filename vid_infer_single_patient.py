import torch
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import glob
import re
import argparse
from pathlib import Path
import json

# Model imports
from src.network.conv_based.CMUNet import CMUNet
from src.network.conv_based.U_Net import U_Net
from src.network.conv_based.AttU_Net import AttU_Net
from src.network.conv_based.UNeXt import UNext
from src.network.conv_based.UNetplus import ResNet34UnetPlus
from src.network.conv_based.UNet3plus import UNet3plus
from src.network.conv_based.CMUNeXt import cmunext
from src.network.transfomer_based.transformer_based_network import get_transformer_based_model


# ------------------------ Utils ------------------------

def seed_torch(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def ckpt_to_tag(ckpt_path: str) -> str:
    """체크포인트 파일명(확장자 제거)을 태그로 사용"""
    return Path(ckpt_path).stem
# 람다 값 추출 유틸: loss_lambdas.json 우선, 없으면 폴더명 파싱
def extract_lambda_suffix(ckpt_path: str) -> str:
    ckpt_dir = Path(ckpt_path).parent
    json_path = ckpt_dir / "loss_lambdas.json"
    # Values possibly available
    seg = kd = con = temp = None
    kdS = kdA = None
    if json_path.exists():
        try:
            with open(json_path, "r") as f:
                data = json.load(f)
            # Preferred keys for new runs (siam): seg/kd/con/temp
            seg = data.get("seg_lam")
            kd = data.get("kd_lam")
            con = data.get("con_lam")
            temp = data.get("temp")
            # Backward-compat keys
            if seg is None and kd is None and con is None:
                kdS = data.get("kd_strong_lam")
                kdA = data.get("kd_anchor_lam")
                con = data.get("con_lam")
        except Exception:
            pass

    # If JSON missing or partial, try parsing directory name
    if seg is None or kd is None or con is None:
        # New style: seg_*_kd_*_con_*[_temp_*]
        m1 = re.search(r"seg_([^_]+)_kd_([^_]+)_con_([^_]+)(?:_temp_([^_]+))?", ckpt_dir.name)
        if m1:
            seg, kd, con, temp = m1.group(1), m1.group(2), m1.group(3), m1.group(4)
        else:
            # Old style: kdS_*_kdA_*_con_*
            m2 = re.search(r"kdS_([^_]+)_kdA_([^_]+)_con_([^_]+)", ckpt_dir.name)
            if m2:
                kdS, kdA, con = m2.group(1), m2.group(2), m2.group(3)

    def fmt(v):
        if isinstance(v, (int, float)):
            return f"{v:g}"
        return str(v) if v is not None else None

    # Build suffix with preference: new style (seg/kd/con[/temp]) else old style (kdS/kdA/con)
    seg, kd, con, temp, kdS, kdA = fmt(seg), fmt(kd), fmt(con), fmt(temp), fmt(kdS), fmt(kdA)
    if seg and kd and con:
        return f"seg_{seg}_kd_{kd}_con_{con}" + (f"_temp_{temp}" if temp else "")
    if kdS and kdA and con:
        return f"kdS_{kdS}_kdA_{kdA}_con_{con}"
    return ""



# ------------------------ Models ------------------------

def get_model(args):
    """모델을 로드하는 함수; transformer 계열도 전역 parser 없이 호출"""
    if args.model == "CMUNet":
        model = CMUNet(output_ch=args.num_classes).cuda()
    elif args.model == "CMUNeXt":
        model = cmunext(num_classes=args.num_classes).cuda()
    elif args.model == "U_Net":
        model = U_Net(output_ch=args.num_classes).cuda()
    elif args.model == "AttU_Net":
        model = AttU_Net(output_ch=args.num_classes).cuda()
    elif args.model == "UNext":
        model = UNext(output_ch=args.num_classes).cuda()
    elif args.model == "UNetplus":
        model = ResNet34UnetPlus(num_class=args.num_classes).cuda()
    elif args.model == "UNet3plus":
        model = UNet3plus(n_classes=args.num_classes).cuda()
    else:
        model = get_transformer_based_model(
            parser=None,
            model_name=args.model,
            img_size=args.img_size,
            num_classes=args.num_classes,
            in_ch=3
        ).cuda()
    return model


def load_model_weights(model, checkpoint_path):
    """모델 가중치를 로드하는 함수 (map_location/strict=False로 안전성 향상)"""
    if os.path.exists(checkpoint_path):
        map_loc = "cuda" if torch.cuda.is_available() else "cpu"
        state = torch.load(checkpoint_path, map_location=map_loc)
        model.load_state_dict(state, strict=False)  # 키 미스매치 시에도 최대한 로드
        print(f"Model loaded from {checkpoint_path}")
    else:
        print(f"Checkpoint not found: {checkpoint_path}")
        return False
    return True


# ------------------------ Transforms & IO ------------------------

def get_transform(img_size=256):
    """인퍼런스용 transform"""
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(),
        ToTensorV2(),
    ])


def extract_patient_and_frame(filename):
    """파일명에서 환자번호와 프레임 번호를 추출 (예: 62844330_100.png -> (62844330, 100))"""
    match = re.match(r'(\d+)_(\d+)\.png', filename)
    if match:
        return match.group(1), int(match.group(2))
    return None, None


def get_patient_frames(video_dir, target_patient_id):
    """특정 환자의 프레임만 추출"""
    image_files = glob.glob(os.path.join(video_dir, "*.png"))
    if not image_files:
        print(f"No PNG files found in {video_dir}")
        return []
    patient_frames = []
    for image_path in image_files:
        filename = os.path.basename(image_path)
        patient_id, frame_num = extract_patient_and_frame(filename)
        if patient_id == target_patient_id:
            patient_frames.append((frame_num, image_path))
    patient_frames.sort(key=lambda x: x[0])
    return patient_frames

def create_video_from_frames(frames, output_path, fps=30, enforce_size=True):
    """
    - 프레임을 uint8/BGR/고정 해상도로 강제
    - mp4 코덱 여러 개 시도 → 실패 시 .avi(MJPG) 폴백
    - 저장 성공 여부를 True/False로 반환
    """
    if not frames:
        print("[ERROR] No frames to create video")
        return False

    # ---- 규격 통일 ----
    h, w = frames[0].shape[:2]
    fixed = []
    for f in frames:
        # 채널 강제 (1채널→3채널)
        if f.ndim == 2:
            f = cv2.cvtColor(f, cv2.COLOR_GRAY2BGR)
        elif f.shape[2] == 1:
            f = cv2.cvtColor(f, cv2.COLOR_GRAY2BGR)
        # dtype 강제
        if f.dtype != np.uint8:
            f = np.clip(f, 0, 255).astype(np.uint8)
        # 크기 강제
        if enforce_size and (f.shape[0] != h or f.shape[1] != w):
            f = cv2.resize(f, (w, h), interpolation=cv2.INTER_LINEAR)
        fixed.append(f)

    # ---- 코덱 시도 ----
    out_path = Path(output_path)
    ext = out_path.suffix.lower()
    candidates = ["mp4v", "avc1", "H264"] if ext == ".mp4" else ["MJPG", "XVID", "mp4v"]

    for fourcc_name in candidates:
        fourcc = cv2.VideoWriter_fourcc(*fourcc_name)
        writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))
        if writer.isOpened():
            for f in fixed:
                writer.write(f)
            writer.release()
            if out_path.exists() and out_path.stat().st_size > 0:
                print(f"[OK] Video saved ({fourcc_name}): {out_path}")
                return True
        else:
            writer.release()

    # ---- 폴백: .avi + MJPG ----
    fallback = out_path.with_suffix(".avi")
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(str(fallback), fourcc, fps, (w, h))
    if writer.isOpened():
        for f in fixed:
            writer.write(f)
        writer.release()
        if fallback.exists() and fallback.stat().st_size > 0:
            print(f"[OK] Fallback saved (MJPG): {fallback}")
            return True

    print("[ERROR] VideoWriter failed for both MP4 and AVI. "
          "OpenCV FFmpeg 지원 또는 쓰기 권한을 확인하세요.")
    return False


# ------------------------ Inference Core ------------------------

def process_patient_video(patient_id, frames_data, model, transform, device, view_type, num_classes=1):
    """특정 환자의 프레임들을 처리하여 영상 생성"""
    if not frames_data:
        print(f"No frames found for patient {patient_id}")
        return []

    print(f"Processing {len(frames_data)} frames for patient {patient_id}, view: {view_type}")

    video_frames = []
    model.eval()
    with torch.no_grad():
        for _, (frame_num, image_path) in enumerate(tqdm(frames_data, desc=f"Processing patient {patient_id} {view_type}")):
            image = cv2.imread(image_path)
            if image is None:
                print(f"Could not load image: {image_path}")
                continue
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            transformed = transform(image=image)
            input_tensor = transformed['image'].unsqueeze(0).to(device)

            output = model(input_tensor)
            if num_classes == 1:
                output = torch.sigmoid(output)

            pred_mask = output.squeeze().cpu().numpy()
            pred_mask_resized = cv2.resize(pred_mask, (image.shape[1], image.shape[0]))
            pred_mask_binary = (pred_mask_resized > 0.5).astype(np.uint8) * 255

            mask_colored = np.zeros_like(image)
            mask_colored[:, :, 1] = pred_mask_binary  # green channel

            contours, _ = cv2.findContours(pred_mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(mask_colored, contours, -1, (255, 0, 0), 2)  # blue contour

            alpha = 0.4
            result_image = cv2.addWeighted(image, 1 - alpha, mask_colored, alpha, 0)
            result_image = cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR)
            video_frames.append(result_image)

    return video_frames


# ------------------------ Main ------------------------

def main():
    parser = argparse.ArgumentParser(description='Single Patient Video Inference')
    parser.add_argument('--patient_id', default='15213598', type=str, help='Patient ID to process')
    parser.add_argument('--model', type=str, default='U_Net', help='Model name')
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
    
    parser.add_argument('--checkpoint', type=str, default=os.path.join(SCRIPT_DIR, "checkpoint", "siam_kd", "0926", "view_axi_siam_seg_2.0_kd_3.0_con_1.5_temp_4.0", "U_Net_studentA_final.pth"), help='Model checkpoint path')
    parser.add_argument('--video_dir', type=str, default=os.path.join(PROJECT_ROOT, "video_test"), help='Video test directory')
    parser.add_argument('--output_dir', type=str, default=os.path.join(PROJECT_ROOT, "single_patient_results"), help='Output directory')
    parser.add_argument('--img_size', type=int, default=256, help='Image size')
    parser.add_argument('--num_classes', type=int, default=1, help='Number of classes')
    parser.add_argument('--fps', type=int, default=30, help='Video FPS')
    parser.add_argument('--seed', type=int, default=41, help='Random seed')
    parser.add_argument('--view', default='axi', type=str, choices=['axi', 'sag', 'both'], help='View type to process')

    args = parser.parse_args()

    print("=== Single Patient Video Inference 설정 ===")
    print(f"환자 ID: {args.patient_id}")
    print(f"모델: {args.model}")
    print(f"체크포인트: {args.checkpoint}")
    print(f"비디오 디렉토리: {args.video_dir}")
    print(f"출력 디렉토리: {args.output_dir}")
    print(f"이미지 크기: {args.img_size}")
    print(f"클래스 수: {args.num_classes}")
    print(f"영상 FPS: {args.fps}")
    print(f"뷰 타입: {args.view}")
    print(f"시드: {args.seed}\n")

    # 시드 & 디바이스
    seed_torch(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 모델 로드
    class ModelArgs:
        def __init__(self):
            self.model = args.model
            self.img_size = args.img_size
            self.num_classes = args.num_classes

    model_args = ModelArgs()
    model = get_model(model_args)
    if not load_model_weights(model, args.checkpoint):
        return

    # Transform
    transform = get_transform(args.img_size)

    # 출력 루트: <output_dir>/<ckpt_tag>/
    os.makedirs(args.output_dir, exist_ok=True)
    ckpt_tag = ckpt_to_tag(args.checkpoint)           # 가중치 파일명(확장자 제외)
    ckpt_root = os.path.join(args.output_dir, ckpt_tag)
    os.makedirs(ckpt_root, exist_ok=True)

    # 처리할 뷰 타입
    view_types = ['axi', 'sag'] if args.view == 'both' else [args.view]

    saved_paths = []
    for view_type in view_types:
        video_dir = os.path.join(args.video_dir, view_type)
        if not os.path.exists(video_dir):
            print(f"Directory not found: {video_dir}")
            continue

        print(f"\n=== Processing {view_type} view for patient {args.patient_id} ===")

        # 환자의 프레임들
        frames_data = get_patient_frames(video_dir, args.patient_id)
        if not frames_data:
            print(f"No frames found for patient {args.patient_id} in {view_type} view")
            continue

        print(f"Found {len(frames_data)} frames for patient {args.patient_id} in {view_type} view")

        # 인퍼런스
        video_frames = process_patient_video(
            args.patient_id, frames_data, model, transform, device,
            view_type, args.num_classes
        )

        if video_frames:
            # 저장 폴더: <output_dir>/<ckpt_tag>/(both인 경우 view 하위 폴더)
            out_dir = ckpt_root if len(view_types) == 1 else os.path.join(ckpt_root, view_type)
            os.makedirs(out_dir, exist_ok=True)

            # 파일명: <patient_id>_<prev_folder>_<date_folder>_<weight_stem>[_<lambda_suffix>].mp4
            lambda_suffix = extract_lambda_suffix(args.checkpoint)
            ckpt_p = Path(args.checkpoint)
            weight_stem = ckpt_p.stem
            date_folder = ckpt_p.parent.parent.name if ckpt_p.parent and ckpt_p.parent.parent else ""
            prev_folder = ckpt_p.parent.parent.parent.name if ckpt_p.parent and ckpt_p.parent.parent and ckpt_p.parent.parent.parent else ""

            parts = [prev_folder, date_folder, weight_stem]
            prefix = "_".join([p for p in parts if p])
            name = args.patient_id if not prefix else f"{args.patient_id}_{prefix}"
            if lambda_suffix:
                name = f"{name}_{lambda_suffix}"
            video_output_path = os.path.join(out_dir, f"{name}.mp4")
            create_video_from_frames(video_frames, video_output_path, args.fps)
            saved_paths.append(video_output_path)

            print(f"Successfully created video: {video_output_path}")
            print(f"Video contains {len(video_frames)} frames")
        else:
            print(f"Failed to create video for patient {args.patient_id} in {view_type} view")

    if saved_paths:
        print("\n=== 저장된 영상 목록 ===")
        for p in saved_paths:
            print(p)
    else:
        print("\n저장된 영상이 없습니다.")


if __name__ == "__main__":
    main()





