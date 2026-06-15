import os
import argparse
import random
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from torch.utils.data import DataLoader
from src.dataloader.dataset import Prostate_Whole_Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
import src.utils.losses as losses
from src.utils.losses import structure_loss
from src.utils.util import AverageMeter, clip_gradient
from src.utils.metrics import iou_score
from src.utils.metrics_SUN import evaluator as evaluator_SUN

from src.network.conv_based.U_Net import U_Net
from src.network.conv_based.UNetplus import ResNet34UnetPlus
from src.network.conv_based.ACSNet import ACSNet
from src.network.hybrid_based.PraNet import PraNet
from src.network.conv_based.ColonSegNet import ColonSegNet
from src.network.hybrid_based.MedNeXt import MedNeXt
from src.network.conv_based.MSRF_Net import MSRF_Net
from src.network.hybrid_based.TransNetR import TransNetR


def seed_torch(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True  # 변경: False -> True (성능 향상)
    torch.backends.cudnn.deterministic = False  # 변경: True -> False (성능 향상)
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default="U_Net",
                    choices=["U_Net", "UNetplus", "ACSNet", "PraNet", "ColonSegNet", "MedNeXt", "MSRF_Net", "TransNetR"], help='model')
parser.add_argument('--dataset_csv', type=str, default="Prostatae_whole.csv", help='csv')
parser.add_argument('--test_csv', type=str, default="Prostatae_whole.csv", help='csv')
parser.add_argument('--base_lr', type=float, default=0.01, help='segmentation network learning rate')
parser.add_argument('--batch_size', type=int, default=4, help='batch_size per gpu')
parser.add_argument('--epoch', type=int, default=50, help='train epoch')
parser.add_argument('--img_size', type=int, nargs=2, default=[256, 448], help='img size of per batch')
parser.add_argument('--num_classes', type=int, default=1, help='seg num_classes')
parser.add_argument('--seed', type=int, default=1225, help='random seed')
parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'], help='train or test mode')
parser.add_argument('--checkpoint', type=str, default=None, help='checkpoint path for testing')
parser.add_argument('--patience', type=int, default=10, help='early stopping patience')
parser.add_argument('--num_workers', type=int, default=8, help='number of workers for dataloader')  # 추가
parser.add_argument('--prefetch_factor', type=int, default=4, help='prefetch factor for dataloader')  # 추가
parser.add_argument('--amp', action='store_true', help='use automatic mixed precision')  # 추가
args = parser.parse_args()
seed_torch(args.seed)


def get_model(args):
    if args.model == "U_Net":
        model = U_Net(output_ch=args.num_classes).cuda()
    elif args.model == "UNetplus":
        model = ResNet34UnetPlus(num_class=args.num_classes).cuda()
    elif args.model == "ACSNet":
        model = ACSNet(num_classes=args.num_classes).cuda()
    elif args.model == "PraNet":
        model = PraNet().cuda()    
    elif args.model == "ColonSegNet":
        model = ColonSegNet().cuda() 
    elif args.model == "MedNeXt":
        model = MedNeXt().cuda() 
    elif args.model == "MSRF_Net":
        model = MSRF_Net(in_ch=3, num_classes=args.num_classes).cuda()
    elif args.model == "TransNetR":
        model = TransNetR(num_classes=args.num_classes, input_hw=args.img_size).cuda()

    return model

def getDataloader(args):
    img_size = args.img_size
    if args.model == "SwinUnet":
        img_size = [224, 224]

    train_transform = A.Compose([
        A.RandomResizedCrop(img_size[0], img_size[1], scale=(0.9, 1.0)), 
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.GaussianBlur(p=0.3),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1,
                        rotate_limit=10, p=0.5),
        A.Cutout(num_holes=4, max_h_size=16, max_w_size=16, p=0.3),
        A.Normalize(),
        ToTensorV2(),
    ])

    val_transform = A.Compose([
        A.Resize(img_size[0], img_size[1]),
        A.Normalize(),
        ToTensorV2(),
    ])

    df = pd.read_csv(args.dataset_csv)
    # train_df, val_df = train_test_split(df, test_size=0.2, random_state=1225, stratify=df['class'])
    train_df = df
    test_df = pd.read_csv(args.test_csv)

    db_train = Prostate_Whole_Dataset(train_df, transform=train_transform)
    db_val = Prostate_Whole_Dataset(test_df, transform=val_transform)
    db_test = Prostate_Whole_Dataset(test_df, transform=val_transform)

    print("train num:{}, val num:{}, test num:{}".format(len(db_train), len(db_val), len(db_test)))

    # 최적화된 DataLoader 설정
    trainloader = DataLoader(
        db_train, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=args.num_workers,  # 조정 가능
        pin_memory=True,
        persistent_workers=True,  # 추가: worker 재사용
        prefetch_factor=args.prefetch_factor  # 추가: prefetch
    )
    valloader = DataLoader(
        db_val, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=args.prefetch_factor
    )
    testloader = DataLoader(
        db_test, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=args.prefetch_factor
    )

    return trainloader, valloader, testloader


def test(args, testloader, model, criterion):
    """Test the model on test dataset"""
    print("\n" + "="*50)
    print("Starting Test Phase")
    print("="*50)
    
    model.eval()
    
    avg_meters = {
        'test_loss': AverageMeter(),
        'test_iou': AverageMeter(),
        'test_dice': AverageMeter(),
        'test_SE': AverageMeter(),
        'test_PC': AverageMeter(),
        'test_F1': AverageMeter(),
        'test_SP': AverageMeter(),
        'test_ACC': AverageMeter(),


        'test_Sm': AverageMeter(),
        'test_mEm': AverageMeter(),
        'test_wF': AverageMeter(),
        'test_mFm': AverageMeter(),
        'test_mD': AverageMeter(),
        'test_mS': AverageMeter(),
    }
    
    test_pbar = tqdm(testloader, desc='Testing', position=0)
    
    with torch.no_grad():
        for i_batch, sampled_batch in enumerate(test_pbar):
            img_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            img_batch, label_batch = img_batch.cuda(non_blocking=True), label_batch.cuda(non_blocking=True)
            
            # Mixed Precision for inference
            if args.model == "PraNet":
                if args.amp:
                    with torch.amp.autocast('cuda'):
                        _, _, _, out_res = model(img_batch)
                        output = F.interpolate(out_res, size=args.img_size[0], mode='bilinear', align_corners=False)
                        loss = structure_loss(output, label_batch)
                else:
                    _, _, _, out_res = model(img_batch)
                    output = F.interpolate(out_res, size=args.img_size[0], mode='bilinear', align_corners=False)
                    loss = structure_loss(output, label_batch)
            else:
                if args.amp:
                    with torch.amp.autocast('cuda'):
                        output = model(img_batch)
                        loss = criterion(output, label_batch)
                else:
                    output = model(img_batch)
                    loss = criterion(output, label_batch)

            iou, dice, SE, PC, F1, SP, ACC = iou_score(output, label_batch)
            sun_results = evaluator_SUN(label_batch, output, metrics=["Smeasure", "meanEm", "wFmeasure", "meanFm", "meanDice", "meanSen"])
            
            avg_meters['test_loss'].update(loss.item(), img_batch.size(0))
            avg_meters['test_iou'].update(iou, img_batch.size(0))
            avg_meters['test_dice'].update(dice, img_batch.size(0))
            avg_meters['test_SE'].update(SE, img_batch.size(0))
            avg_meters['test_PC'].update(PC, img_batch.size(0))
            avg_meters['test_F1'].update(F1, img_batch.size(0))
            avg_meters['test_SP'].update(SP, img_batch.size(0))
            avg_meters['test_ACC'].update(ACC, img_batch.size(0))

            avg_meters['test_Sm'].update(sun_results['Smeasure'], img_batch.size(0))
            avg_meters['test_mEm'].update(sun_results['meanEm'], img_batch.size(0))
            avg_meters['test_wF'].update(sun_results['wFmeasure'], img_batch.size(0))
            avg_meters['test_mFm'].update(sun_results['meanFm'], img_batch.size(0))
            avg_meters['test_mD'].update(sun_results['meanDice'], img_batch.size(0))
            avg_meters['test_mS'].update(sun_results['meanSen'], img_batch.size(0))

            test_pbar.set_postfix({
                'loss': f'{avg_meters["test_loss"].avg:.4f}',
                'iou': f'{avg_meters["test_iou"].avg:.4f}',
                'dice': f'{avg_meters["test_dice"].avg:.4f}'
            })
    
    print("\n" + "="*50)
    print("Test Results Summary")
    print("="*50)
    print(f"Test Loss:      {avg_meters['test_loss'].avg:.4f}")
    print(f"Test IoU:       {avg_meters['test_iou'].avg:.4f}")
    print(f"Test Dice:      {avg_meters['test_dice'].avg:.4f}")
    print(f"Test Sensitivity (SE): {avg_meters['test_SE'].avg:.4f}")
    print(f"Test Precision (PC):   {avg_meters['test_PC'].avg:.4f}")
    print(f"Test F1 Score:         {avg_meters['test_F1'].avg:.4f}")
    print(f"Test Specificity (SP): {avg_meters['test_SP'].avg:.4f}")
    print(f"Test Accuracy (ACC):   {avg_meters['test_ACC'].avg:.4f}")
    print("="*50)
    print("SUN Metrics")
    print(f"Test S-measure:        {avg_meters['test_Sm'].avg:.4f}")
    print(f"Test mean E-measure:   {avg_meters['test_mEm'].avg:.4f}")
    print(f"Test weighted F-measure: {avg_meters['test_wF'].avg:.4f}")
    print(f"Test mean F-measure:    {avg_meters['test_mFm'].avg:.4f}")
    print(f"Test mean Dice:         {avg_meters['test_mD'].avg:.4f}")
    print(f"Test mean Sensitivity: {avg_meters['test_mS'].avg:.4f}")
    print("="*50 + "\n")
    
    return avg_meters

def train(args):
    """Training function with optimization"""
    base_lr = args.base_lr

    trainloader, valloader, testloader = getDataloader(args=args)

    model = get_model(args)
    
    # Compile model (PyTorch 2.0+)
    if hasattr(torch, 'compile'):
        model = torch.compile(model)
        print("Model compiled with torch.compile()")

    optimizer = optim.AdamW(model.parameters(), lr=base_lr, weight_decay=1e-4)
    criterion = losses.__dict__['BCEDiceLoss']().cuda()
    
    # Mixed Precision Training 설정
    scaler = torch.amp.GradScaler('cuda') if args.amp else None
    if args.amp:
        print("Using Automatic Mixed Precision (AMP)")

    print("{} iterations per epoch".format(len(trainloader)))
    best_dsc = 0
    iter_num = 0
    patience = 0
    max_epoch = args.epoch

    max_iterations = len(trainloader) * max_epoch
    
    epoch_pbar = tqdm(range(max_epoch), desc='Training Progress', position=0)
    
    for epoch_num in epoch_pbar:
        model.train()
        avg_meters = {'loss': AverageMeter(),
                      'iou': AverageMeter(),
                      'dsc': AverageMeter(),
                      'val_loss': AverageMeter(),
                      'val_iou': AverageMeter(),
                      'val_dsc': AverageMeter(),
                      'val_SE': AverageMeter(),
                      'val_PC': AverageMeter(),
                      'val_F1': AverageMeter(),
                      'val_ACC': AverageMeter()}

        train_pbar = tqdm(trainloader, desc=f'Epoch {epoch_num+1}/{max_epoch} [Train]', 
                         position=1, leave=False)
        
        for i_batch, sampled_batch in enumerate(train_pbar):

            img_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            img_batch, label_batch = img_batch.cuda(non_blocking=True), label_batch.cuda(non_blocking=True)

            optimizer.zero_grad(set_to_none=True)  # 변경: 더 효율적

            if args.model == "PraNet":
                for rate in [0.75, 1, 1.25]:
                    if args.amp:
                        with torch.amp.autocast('cuda'):
                            # ---- rescale ----
                            trainsize = int(round(args.img_size[0]*rate/32)*32)
                            if rate != 1:
                                images = F.interpolate(img_batch, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                                gts = F.interpolate(label_batch, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                            # ---- forward ----
                            lateral_map_5, lateral_map_4, lateral_map_3, lateral_map_2 = model(images)
                            # ---- loss function ----
                            loss5 = structure_loss(lateral_map_5, gts)
                            loss4 = structure_loss(lateral_map_4, gts)
                            loss3 = structure_loss(lateral_map_3, gts)
                            loss2 = structure_loss(lateral_map_2, gts)
                            loss = loss2 + loss3 + loss4 + loss5    # TODO: try different weights for loss
                        # ---- backward ----
                        loss.backward()
                        clip_gradient(optimizer, 0.5)
                        optimizer.step()
                        if rate == 1: outputs = F.interpolate(lateral_map_2, size=args.img_size[0], mode='bilinear', align_corners=False)

            else:
                # Mixed Precision Training
                if args.amp:
                    with torch.amp.autocast('cuda'):
                        outputs = model(img_batch)
                        loss = criterion(outputs, label_batch)
                    
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    outputs = model(img_batch)
                    loss = criterion(outputs, label_batch)
                    loss.backward()
                    optimizer.step()

            iou, dice, _, _, _, _, _ = iou_score(outputs, label_batch)

            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1
            avg_meters['loss'].update(loss.item(), img_batch.size(0))
            avg_meters['iou'].update(iou, img_batch.size(0))
            avg_meters['dsc'].update(dice, img_batch.size(0))
            
            train_pbar.set_postfix({
                'loss': f'{avg_meters["loss"].avg:.4f}',
                'iou': f'{avg_meters["iou"].avg:.4f}',
                'dsc': f'{avg_meters["dsc"].avg:.4f}',
                'lr': f'{lr_:.6f}'
            })

        # Validation
        model.eval()
        val_pbar = tqdm(valloader, desc=f'Epoch {epoch_num+1}/{max_epoch} [Val]', 
                       position=1, leave=False)
        
        with torch.no_grad():
            for i_batch, sampled_batch in enumerate(val_pbar):
                img_batch, label_batch = sampled_batch['image'], sampled_batch['label']
                img_batch, label_batch = img_batch.cuda(non_blocking=True), label_batch.cuda(non_blocking=True)

                if args.model == "PraNet":
                    _, _, _, out_res = model(img_batch)
                    output = F.interpolate(out_res, size=args.img_size[0], mode='bilinear', align_corners=False)
                    loss = structure_loss(output, label_batch)
                else:
                    output = model(img_batch)
                    loss = criterion(output, label_batch)
                iou, dice, SE, PC, F1, _, ACC = iou_score(output, label_batch)
                avg_meters['val_loss'].update(loss.item(), img_batch.size(0))
                avg_meters['val_iou'].update(iou, img_batch.size(0))
                avg_meters['val_dsc'].update(dice, img_batch.size(0))
                avg_meters['val_SE'].update(SE, img_batch.size(0))
                avg_meters['val_PC'].update(PC, img_batch.size(0))
                avg_meters['val_F1'].update(F1, img_batch.size(0))
                avg_meters['val_ACC'].update(ACC, img_batch.size(0))
                
                val_pbar.set_postfix({
                    'val_loss': f'{avg_meters["val_loss"].avg:.4f}',
                    'val_iou': f'{avg_meters["val_iou"].avg:.4f}'
                })

        epoch_pbar.set_postfix({
            'train_loss': f'{avg_meters["loss"].avg:.4f}',
            'train_iou': f'{avg_meters["iou"].avg:.4f}',
            'val_loss': f'{avg_meters["val_loss"].avg:.4f}',
            'val_iou': f'{avg_meters["val_iou"].avg:.4f}',
            'val_dsc': f'{avg_meters["val_dsc"].avg:.4f}',
            'best_dsc': f'{best_dsc:.4f}'
        })

        tqdm.write(f'\nEpoch [{epoch_num+1}/{max_epoch}] Summary:')
        tqdm.write(f'  Train - Loss: {avg_meters["loss"].avg:.4f}, IoU: {avg_meters["iou"].avg:.4f}, DSC: {avg_meters["dsc"].avg:.4f}')
        tqdm.write(f'  Val   - Loss: {avg_meters["val_loss"].avg:.4f}, IoU: {avg_meters["val_iou"].avg:.4f}, DSC: {avg_meters["val_dsc"].avg:.4f}, '
                  f'SE: {avg_meters["val_SE"].avg:.4f}, PC: {avg_meters["val_PC"].avg:.4f}, '
                  f'F1: {avg_meters["val_F1"].avg:.4f}, ACC: {avg_meters["val_ACC"].avg:.4f}')

        if avg_meters['val_dsc'].avg > best_dsc:
            if not os.path.isdir("./checkpoint"):
                os.makedirs("./checkpoint")
            torch.save(model.state_dict(), 'checkpoint/{}_model.pth'.format(args.model))
            best_dsc = avg_meters['val_dsc'].avg
            patience = 0
            tqdm.write("  => Saved best model!")
        else:
            patience += 1
            
        if patience > args.patience:
            tqdm.write(f"  => Early stopping triggered (patience={patience})")
            break

    print("\n" + "="*50)
    print("Training Finished! Loading best model for testing...")
    print("="*50)
    
    best_model_path = 'checkpoint/{}_model.pth'.format(args.model)
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path))
        print(f"Loaded best model from {best_model_path}")
        test(args, testloader, model, criterion)
    else:
        print("Best model not found. Testing with final model...")
        test(args, testloader, model, criterion)

    return "Training and Testing Finished!"

def main(args):
    if args.mode == 'train':
        return train(args)
    elif args.mode == 'test':
        trainloader, valloader, testloader = getDataloader(args=args)
        model = get_model(args)
        
        # Compile model (PyTorch 2.0+)
        if hasattr(torch, 'compile'):
            model = torch.compile(model)
            print("Model compiled with torch.compile()")
        
        criterion = losses.__dict__['BCEDiceLoss']().cuda()
        
        checkpoint_path = args.checkpoint if args.checkpoint else f'checkpoint/{args.model}_model.pth'
        
        if not os.path.exists(checkpoint_path):
            print(f"Error: Checkpoint not found at {checkpoint_path}")
            return
        
        print(f"Loading checkpoint from {checkpoint_path}")
        try:
            model.load_state_dict(torch.load(checkpoint_path))
        except:
            model.load_state_dict(torch.load(checkpoint_path)['model_state_dict'])

        if args.amp:
            print("Using Automatic Mixed Precision (AMP) for testing")
        
        test(args, testloader, model, criterion)
        return "Testing Finished!"


if __name__ == "__main__":
    main(args)

"""

Training
CUDA_VISIBLE_DEVICES=0 python main.py --model U_Net --dataset_csv csvs/TRUS_V_train_df.csv --test_csv csvs/TRUS_V_test_df.csv --base_lr 3e-4 --epoch 50 --batch_size 32 --mode train --prefetch_factor 4 --amp --img_size 256 256
CUDA_VISIBLE_DEVICES=0 python main.py --model UNetplus --dataset_csv csvs/TRUS_V_train_df.csv --test_csv csvs/TRUS_V_test_df.csv --base_lr 3e-4 --epoch 50 --batch_size 32 --mode train --prefetch_factor 4 --amp --img_size 256 256
CUDA_VISIBLE_DEVICES=0 python main.py --model ACSNet --dataset_csv csvs/TRUS_V_train_df.csv --test_csv csvs/TRUS_V_test_df.csv --base_lr 3e-4 --epoch 50 --batch_size 32 --mode train --prefetch_factor 4 --amp --img_size 256 256
CUDA_VISIBLE_DEVICES=3 python main.py --model PraNet --dataset_csv csvs/TRUS_V_train_df.csv --test_csv csvs/TRUS_V_test_df.csv --base_lr 3e-4 --epoch 50 --batch_size 32 --mode train --prefetch_factor 4 --amp --img_size 256 256
CUDA_VISIBLE_DEVICES=3 python main.py --model ColonSegNet --dataset_csv csvs/TRUS_V_train_df.csv --test_csv csvs/TRUS_V_test_df.csv --base_lr 3e-4 --epoch 50 --batch_size 32 --mode train --prefetch_factor 4 --amp --img_size 256 256
CUDA_VISIBLE_DEVICES=3 python main.py --model MedNeXt --dataset_csv csvs/TRUS_V_train_df.csv --test_csv csvs/TRUS_V_test_df.csv --base_lr 1e-5 --epoch 50 --batch_size 32 --mode train --prefetch_factor 4 --img_size 256 256
CUDA_VISIBLE_DEVICES=3 python main.py --model MSRF_Net --dataset_csv csvs/TRUS_V_train_df.csv --test_csv csvs/TRUS_V_test_df.csv --base_lr 3e-4 --epoch 50 --batch_size 32 --mode train --prefetch_factor 4 --amp --img_size 256 256
CUDA_VISIBLE_DEVICES=0 python main.py --model TransNetR --dataset_csv csvs/TRUS_V_train_df.csv --test_csv csvs/TRUS_V_test_df.csv --base_lr 3e-4 --epoch 50 --batch_size 32 --mode train --prefetch_factor 4 --amp --img_size 256 256

CUDA_VISIBLE_DEVICES=3 python main.py --model TransNetR --dataset_csv csvs/df.csv --test_csv csvs/df_easy_unseen.csv --base_lr 3e-4 --epoch 50 --batch_size 32 --mode train --prefetch_factor 4 --amp --img_size 256 256


CUDA_VISIBLE_DEVICES= python main.py --model SwinUnet --dataset_csv csvs/df.csv --test_csv csvs/df_easy_seen.csv --base_lr 3e-4 --epoch 50 --batch_size 32 --mode train  --prefetch_factor 4 --amp

Training 2
CUDA_VISIBLE_DEVICES=2 python main.py --model ACSNet --dataset_csv csvs/df.csv --test_csv csvs/df_easy_seen.csv --base_lr 3e-4 --epoch 50 --batch_size 32 --mode train --prefetch_factor 4 --amp

CUDA_VISIBLE_DEVICES=1 python main.py --model UNext --dataset_csv csvs/df.csv --test_csv csvs/df_easy_seen.csv --base_lr 3e-4 --epoch 50 --batch_size 32 --mode train--prefetch_factor 4 --amp

Testing only
CUDA_VISIBLE_DEVICES=0 python main.py --model U_Net --dataset_csv csvs/df.csv --test_csv csvs/df_easy_unseen.csv --batch_size 32 --mode test --checkpoint checkpoint/U_Net_model.pth --amp
CUDA_VISIBLE_DEVICES=1 python main.py --model AttU_Net --dataset_csv csvs/df.csv --test_csv csvs/df_easy_unseen.csv --batch_size 32 --mode test --checkpoint checkpoint/AttU_Net_model.pth --amp
CUDA_VISIBLE_DEVICES=1 python main.py --model ACSNet --dataset_csv csvs/df.csv --test_csv csvs/df_easy_unseen.csv --batch_size 32 --mode test --checkpoint checkpoint/ACSNet_model.pth --amp
CUDA_VISIBLE_DEVICES=3 python main.py --model TransNetR --dataset_csv csvs/df.csv --test_csv csvs/df_easy_unseen.csv --batch_size 32 --mode test --checkpoint checkpoint/TransNetR_model.pth --amp
CUDA_VISIBLE_DEVICES=4 python main.py --model UNetplus --dataset_csv csvs/df.csv --test_csv csvs/df_easy_unseen.csv --batch_size 32 --mode test --checkpoint checkpoint/UNetplus_model.pth --amp
CUDA_VISIBLE_DEVICES=5 python main.py --model UNext --dataset_csv csvs/df.csv --test_csv csvs/df_easy_unseen.csv --batch_size 32 --mode test --checkpoint checkpoint/UNext_model.pth --amp
CUDA_VISIBLE_DEVICES=6 python main.py --model MSRF_Net --dataset_csv csvs/df.csv --test_csv csvs/df_easy_unseen.csv --batch_size 32 --mode test --checkpoint checkpoint/MSRF_Net_model.pth --amp
CUDA_VISIBLE_DEVICES=7 python main.py --model CMUNeXt --dataset_csv csvs/df.csv --test_csv csvs/df_easy_unseen.csv --batch_size 32 --mode test --checkpoint checkpoint/CMUNeXt_model.pth --amp

CUDA_VISIBLE_DEVICES=0 python main.py --model ACSNet --dataset_csv csvs/df.csv --test_csv csvs/df_easy_unseen.csv --batch_size 32 --mode test --checkpoint /KONERSTON/Temporal/Temporal/checkpoint/flip_kd/ACSNet_Prototype_c4/ACSNet_best.pth --amp
CUDA_VISIBLE_DEVICES=1 python main.py --model ACSNet --dataset_csv csvs/df.csv --test_csv csvs/df_easy_seen.csv --batch_size 32 --mode test --checkpoint /KONERSTON/Temporal/Temporal/checkpoint/flip_kd/ACSNet_Prototype_c4/ACSNet_best.pth --amp
CUDA_VISIBLE_DEVICES=2 python main.py --model ACSNet --dataset_csv csvs/df.csv --test_csv csvs/df_hard_unseen.csv --batch_size 32 --mode test --checkpoint /KONERSTON/Temporal/Temporal/checkpoint/flip_kd/ACSNet_Prototype_c4/ACSNet_best.pth --amp
CUDA_VISIBLE_DEVICES=3 python main.py --model ACSNet --dataset_csv csvs/df.csv --test_csv csvs/df_hard_seen.csv --batch_size 32 --mode test --checkpoint /KONERSTON/Temporal/Temporal/checkpoint/flip_kd/ACSNet_Prototype_c4/ACSNet_best.pth --amp

Long Training for TransNetR
CUDA_VISIBLE_DEVICES=7 python main.py --model TransNetR --dataset_csv csvs/df.csv --test_csv csvs/df_easy_seen.csv --base_lr 1e-4 --epoch 50 --batch_size 32 --mode train --prefetch_factor 4 --amp --patience 50 --epoch 500


CUDA_VISIBLE_DEVICES=1 python main.py --model ACSNet --dataset_csv csvs/df.csv --test_csv csvs/df_easy_unseen.csv --batch_size 32 --mode test --checkpoint /home/work/DY_JH/DY/Temporal/Temporal/checkpoint/flip_kd/ACSNet_Prototype_a8/ACSNet_best.pth --amp
CUDA_VISIBLE_DEVICES=0 python main.py --model ACSNet --dataset_csv csvs/df.csv --test_csv csvs/df_easy_unseen.csv --batch_size 32 --mode test --checkpoint /home/work/DY_JH/DY/Temporal/Temporal/checkpoint/flip_kd/ACSNet_Prototype_a2/ACSNet_best.pth --amp
CUDA_VISIBLE_DEVICES=0 python main.py --model ACSNet --dataset_csv csvs/df.csv --test_csv csvs/df_easy_unseen.csv --batch_size 32 --mode test --checkpoint /home/work/DY_JH/DY/Temporal/Temporal/checkpoint/flip_kd/ACSNet_Prototype_c4/ACSNet_best.pth --amp
CUDA_VISIBLE_DEVICES=1 python main.py --model ACSNet --dataset_csv csvs/df.csv --test_csv csvs/df_easy_unseen.csv --batch_size 32 --mode test --checkpoint /home/work/DY_JH/DY/Temporal/Temporal/checkpoint/flip_kd/ACSNet_Prototype_a8/ACSNet_best.pth --amp


CUDA_VISIBLE_DEVICES=2 python main.py --model ACSNet --dataset_csv csvs/TRUS_V_train_df.csv --test_csv csvs/TRUS_V_test_df.csv --batch_size 32 --mode test --checkpoint /home/work/DY_JH/DY/Temporal/Temporal/checkpoint/flip_kd/ACSNet_Prototype_t1/ACSNet_final.pth --amp --img_size 256 256

"""