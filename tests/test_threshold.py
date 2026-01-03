import numpy as np
from ntprobnp_dss.threshold import choose_threshold_for_sensitivity

def test_threshold_meets_sensitivity():
    y = np.array([1,1,1,0,0,0])
    p = np.array([0.9,0.8,0.7,0.6,0.2,0.1])
    res = choose_threshold_for_sensitivity(y, p, 0.66)
    assert res.sensitivity >= 0.66
