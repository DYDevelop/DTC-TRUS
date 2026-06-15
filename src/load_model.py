import os
import torch
from src.network.conv_based.U_Net import U_Net
from src.network.conv_based.UNetplus import ResNet34UnetPlus
from src.network.conv_based.ACSNet import ACSNet
from src.network.hybrid_based.PraNet import PraNet
from src.network.conv_based.ColonSegNet import ColonSegNet
from src.network.hybrid_based.MedNeXt import MedNeXt
from src.network.conv_based.MSRF_Net import MSRF_Net
from src.network.hybrid_based.TransNetR import TransNetR

def get_model(args):
    """사전 학습된 2D 세그멘테이션 모델 로드"""
    if args.model == "U_Net":
        model = U_Net(output_ch=args.num_classes).cuda()
    elif args.model == "UNetplus":
        model = ResNet34UnetPlus(num_class=args.num_classes).cuda()
    elif args.model == "ACSNet":
        model = ACSNet(num_classes=args.num_classes, feature_return=args.use_prototype).cuda()
    elif args.model == "PraNet":
        model = PraNet().cuda()    
    elif args.model == "ColonSegNet":
        model = ColonSegNet().cuda() 
    elif args.model == "MedNeXt":
        model = MedNeXt().cuda() 
    elif args.model == "MSRF_Net":
        model = MSRF_Net(in_ch=3, num_classes=args.num_classes).cuda()
    elif args.model == "TransNetR":
        model = TransNetR(num_classes=args.num_classes, input_hw=args.img_size, feature_return=args.use_prototype).cuda()
        # 사전 학습된 2D 세그멘테이션 가중치 로드
    if os.path.exists(args.checkpoint):
        try:
            model.load_state_dict(torch.load(args.checkpoint))
        except:
            print("Using Compiled Model")
            model = torch.compile(model)
            model.load_state_dict(torch.load(args.checkpoint))
        print(f"Loaded pretrained 2D segmentation model: {args.checkpoint}")
    else:
        print(f"Warning: Pretrained model not found: {args.checkpoint}")
    
    return model