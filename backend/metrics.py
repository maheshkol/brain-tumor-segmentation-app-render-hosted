import numpy as np

def dice(pred, gt):
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    inter = (pred & gt).sum()
    return 2 * inter / (pred.sum() + gt.sum() + 1e-6)

def iou(pred, gt):
    inter = (pred & gt).sum()
    union = (pred | gt).sum()
    return inter / (union + 1e-6)
