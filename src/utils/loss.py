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

def focal_loss(pred=None, target=None, alpha=.25, gamma=2) : 
    BCE = F.binary_cross_entropy_with_logits(pred, target)
    BCE_EXP = torch.exp(-BCE)
    loss = alpha * (1-BCE_EXP)**gamma * BCE
    return loss

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

def bce_loss(pred, target):
    return F.binary_cross_entropy_with_logits(pred, target)