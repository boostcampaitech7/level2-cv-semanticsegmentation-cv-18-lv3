import torch.nn.functional as F
import torch

def calc_loss_bce_dice(pred=None, target=None, bce_weight=0.5):
    bce = F.binary_cross_entropy_with_logits(pred, target)
    pred = F.sigmoid(pred)
    dice = dice_loss(pred, target)
    loss = bce * bce_weight + dice * (1 - bce_weight)
    return loss

def dice_loss(pred=None, target=None, smooth = 1.):
    pred = pred.contiguous()
    target = target.contiguous()   
    intersection = (pred * target).sum(dim=2).sum(dim=2)
    loss = (1 - ((2. * intersection + smooth) / (pred.sum(dim=2).sum(dim=2) +   target.sum(dim=2).sum(dim=2) + smooth)))
    return loss.mean()

def focal_loss(pred=None, target=None, alpha=0.25, gamma=2.0):
    """Focal Loss: L_fl = -(1 - p_t)^γ * log(p_t)"""
    pred_prob = torch.sigmoid(pred)
    pt = pred_prob * target + (1 - pred_prob) * (1 - target)
    loss = -alpha * (1 - pt) ** gamma * torch.log(pt + 1e-8)  
    
    return loss.mean()


def focal_dice_loss(pred=None, target=None, focal_weight = 0.5):
    focal = focal_loss(pred, target)
    pred = F.sigmoid(pred)
    dice = dice_loss(pred, target)
    loss = focal * focal_weight + dice * (1 - focal_weight)
    return loss
    
def structure_loss(pred=None, target=None):
    weit = 1 + 5 * torch.abs(F.avg_pool2d(target, kernel_size=31, stride=1, padding=15) - target)
    wbce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))
    pred = torch.sigmoid(pred)
    inter = ((pred * target) * weit).sum(dim=(2, 3))
    union = ((pred + target) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()

def multiscale_structure_loss(pred=None, target=None):
    loss = 0
    pred0, pred1, pred2 = pred
    loss0 = structure_loss(pred0, target)
    loss1 = structure_loss(pred1, target)
    loss2 = structure_loss(pred2, target)
    loss = loss0 + loss1 + loss2
    return loss

def cross_entropy_loss(pred=None, target=None):
    return F.cross_entropy(pred, target)

def bce_loss(pred=None, target=None):
    return F.binary_cross_entropy_with_logits(pred, target)

def ms_ssim_loss(pred=None, target=None, C1=0.01**2, C2=0.03**2):
        """수식을 못 쓰겟음"""
        pred = torch.sigmoid(pred)
        
        mu_pred = F.avg_pool2d(pred, kernel_size=11, stride=1, padding=5)
        mu_target = F.avg_pool2d(target, kernel_size=11, stride=1, padding=5)
        
        sigma_pred = F.avg_pool2d(pred**2, kernel_size=11, stride=1, padding=5) - mu_pred**2
        sigma_target = F.avg_pool2d(target**2, kernel_size=11, stride=1, padding=5) - mu_target**2
        sigma_pred_target = F.avg_pool2d(pred*target, kernel_size=11, stride=1, padding=5) - mu_pred*mu_target
        
        numerator = (2*mu_pred*mu_target + C1) * (2*sigma_pred_target + C2)
        denominator = (mu_pred**2 + mu_target**2 + C1) * (sigma_pred**2 + sigma_target**2 + C2)
        ssim = numerator / (denominator + 1e-8)
        
        return 1 - ssim.mean()

def iou_loss(pred=None, target=None, smooth=1.):
    """dice랑 비슷하지만 분모가 합이냐 합집합이냐 차이정도 있음"""
    pred = pred.contiguous()
    target = target.contiguous()
    intersection = (pred * target).sum(dim=2).sum(dim=2)
    
    union = pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) - intersection
    
    iou = (intersection + smooth) / (union + smooth)
    loss = 1 - iou
    return loss.mean()

def unet3p_loss(pred, target, gamma=2.0, beta=1.0, C1=0.01**2, C2=0.03**2):
    """Total Segmentation Loss: L_seg = L_fl + L_ms-ssim + L_iou"""
    l_fl = focal_loss(pred, target, gamma)
    l_ms_ssim = ms_ssim_loss(pred, target, beta, C1, C2)
    l_iou = iou_loss(pred, target)
    return l_fl + l_ms_ssim + l_iou