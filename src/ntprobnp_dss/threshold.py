from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple
import numpy as np

@dataclass(frozen=True)
class ThresholdResult:
    threshold: float
    sensitivity: float
    specificity: float
    ppv: float
    npv: float

def _confusion(y_true: np.ndarray, y_prob: np.ndarray, thr: float) -> Tuple[int,int,int,int]:
    y_pred = (y_prob >= thr).astype(int)
    tp = int(((y_pred==1) & (y_true==1)).sum())
    fp = int(((y_pred==1) & (y_true==0)).sum())
    tn = int(((y_pred==0) & (y_true==0)).sum())
    fn = int(((y_pred==0) & (y_true==1)).sum())
    return tp, fp, tn, fn

def _metrics(tp:int, fp:int, tn:int, fn:int) -> Tuple[float,float,float,float]:
    sens = tp/(tp+fn) if (tp+fn)>0 else float("nan")
    spec = tn/(tn+fp) if (tn+fp)>0 else float("nan")
    ppv  = tp/(tp+fp) if (tp+fp)>0 else float("nan")
    npv  = tn/(tn+fn) if (tn+fn)>0 else float("nan")
    return sens, spec, ppv, npv

def choose_threshold_for_sensitivity(y_true: np.ndarray, y_prob: np.ndarray, min_sens: float) -> ThresholdResult:
    candidates = np.unique(y_prob)
    candidates.sort()

    best = None
    for thr in candidates[::-1]:
        tp, fp, tn, fn = _confusion(y_true, y_prob, float(thr))
        sens, spec, ppv, npv = _metrics(tp, fp, tn, fn)
        if np.isnan(sens):
            continue
        if sens >= min_sens:
            if best is None or spec > best.specificity or (spec == best.specificity and thr > best.threshold):
                best = ThresholdResult(float(thr), float(sens), float(spec), float(ppv), float(npv))

    if best is None:
        # fallback: 0.5
        tp, fp, tn, fn = _confusion(y_true, y_prob, 0.5)
        sens, spec, ppv, npv = _metrics(tp, fp, tn, fn)
        best = ThresholdResult(0.5, float(sens), float(spec), float(ppv), float(npv))

    return best
