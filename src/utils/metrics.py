import torch

def get_accuracy(SR, GT, threshold=0.5):
    SR = SR > threshold
    GT = GT == torch.max(GT)
    corr = torch.sum(SR == GT)
    tensor_size = SR.size(0)*SR.size(1)*SR.size(2)*SR.size(3)
    acc = float(corr)/float(tensor_size)
    return acc

def get_sensitivity(SR, GT, threshold=0.5):
    # Sensitivity == Recall
    SE = 0
    SR = SR > threshold
    GT = GT == torch.max(GT)
    TP = ((SR == 1).byte() + (GT == 1).byte()) == 2
    FN = ((SR == 0).byte() + (GT == 1).byte()) == 2
    SE = float(torch.sum(TP))/(float(torch.sum(TP+FN)) + 1e-6)
    return SE

def get_specificity(SR, GT, threshold=0.5):
    SP = 0
    SR = SR > threshold
    GT = GT == torch.max(GT)
    TN = ((SR == 0).byte() + (GT == 0).byte()) == 2
    FP = ((SR == 1).byte() + (GT == 0).byte()) == 2
    SP = float(torch.sum(TN))/(float(torch.sum(TN+FP)) + 1e-6)
    return SP

def get_precision(SR, GT, threshold=0.5):
    PC = 0
    SR = SR > threshold
    GT = GT== torch.max(GT)
    TP = ((SR == 1).byte() + (GT == 1).byte()) == 2
    FP = ((SR == 1).byte() + (GT == 0).byte()) == 2
    PC = float(torch.sum(TP))/(float(torch.sum(TP+FP)) + 1e-6)
    return PC

def iou_score(output, target):
    smooth = 1e-5

    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
    output_ = output > 0.5
    target_ = target > 0.5
    
    intersection = (output_ & target_).sum()
    union = (output_ | target_).sum()
    iou = (intersection + smooth) / (union + smooth)
    dice = (2 * iou) / (iou+1)
    
    output_ = torch.tensor(output_)
    target_ = torch.tensor(target_)
    SE = get_sensitivity(output_, target_, threshold=0.5)
    PC = get_precision(output_, target_, threshold=0.5)
    SP = get_specificity(output_, target_, threshold=0.5)
    ACC = get_accuracy(output_, target_, threshold=0.5)
    F1 = 2*SE*PC/(SE+PC + 1e-6)
    return iou, dice, SE, PC, F1, SP, ACC

def dice_coef(output, target):
    smooth = 1e-5
    output = torch.sigmoid(output).view(-1).data.cpu().numpy()
    target = target.view(-1).data.cpu().numpy()
    intersection = (output * target).sum()
    return (2. * intersection + smooth) / \
        (output.sum() + target.sum() + smooth)

def multiclass_metrics(output, target, threshold=0.5):
    """
    output: 모델 출력 (BxCxHxW)
    target: 실제 정답 맵 (BxCxHxW)
    """
    total_metrics = []
    smooth = 1e-5

    for channel in range(output.size(1)):
        metrics = {
                    "iou": [],
                    "dsc": [],
                    "sensitivity": [],
                    "specificity": [],
                    "precision": [],
                    "accuracy": [],
                    "f1_score": []
                }

        output_cls = (output[:, channel, :, :] > threshold).bool()
        target_cls = (target[:, channel, :, :] > 0).bool()
        
        intersection = (output_cls & target_cls).sum().item()
        union = (output_cls | target_cls).sum().item()
        
        iou = (intersection + smooth) / (union + smooth)
        dice = (2 * intersection + smooth) / (output_cls.sum().item() + target_cls.sum().item() + smooth)
        
        SR = output_cls.float()
        GT = target_cls.float()
        
        TP = torch.sum((SR == 1) & (GT == 1)).item()
        TN = torch.sum((SR == 0) & (GT == 0)).item()
        FP = torch.sum((SR == 1) & (GT == 0)).item()
        FN = torch.sum((SR == 0) & (GT == 1)).item()
        
        SE = TP / (TP + FN + 1e-6)  # Sensitivity
        SP = TN / (TN + FP + 1e-6)  # Specificity
        PC = TP / (TP + FP + 1e-6)  # Precision
        ACC = (TP + TN) / (TP + TN + FP + FN + 1e-6)  # Accuracy
        F1 = 2 * SE * PC / (SE + PC + 1e-6)  # F1 Score

        metrics['iou'].append(iou)
        metrics['dsc'].append(dice)
        metrics["sensitivity"].append(SE)
        metrics["specificity"].append(SP)
        metrics["precision"].append(PC)
        metrics["accuracy"].append(ACC)
        metrics["f1_score"].append(F1)

        total_metrics.append(metrics)
    
    return total_metrics