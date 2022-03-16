'''
Description: 
Version: 1.0
Author: SHAO Nuoya
Date: 2022-03-16 00:19:40
LastEditors: SHAO Nuoya
LastEditTime: 2022-03-16 16:21:40
'''
from single_vasicek import SingleVasicek
from multiple_vasicek import MultipleVasicek
from calibration import SingleCalibrate, MultipleCalibrate
import matplotlib.pyplot as plt
import numpy as np

#***********************Prediction**************************
def single_curve_prediction():
    VS = SingleVasicek()

    ts = np.linspace(0, 1, VS.T)
    ts_test = np.linspace(0, 1, VS.T // 2)

    rts = VS.generate_rt(ts)
    y = VS.Log_Pt(rts)

    MuY_ = VS.mu(ts_test)
    MuY = VS.mu(ts)

    SigmaY_Y = VS.Sigma(ts_test, ts)
    SigmaYY_ = VS.Sigma(ts, ts_test)
    SigmaY_Y_ = VS.Sigma(ts_test, ts_test)
    SigmaYY = VS.Sigma(ts, ts, np.var(y))

    mu_ = MuY_ + SigmaY_Y @ np.linalg.inv(SigmaYY) @ (y - MuY)
    Sigma_ = SigmaY_Y_ - SigmaY_Y @ np.linalg.inv(SigmaYY) @ SigmaYY_
    
    #todo confidence interval to draw
    plt.scatter(ts, y, s=20, label='observation')
    plt.plot(ts_test, mu_, color='red', label='prediction')
    plt.legend()
    #plt.savefig('Single.png')
    plt.show()

#*************************Calibration***************************
def calibrate(curve='single', method='CG'):
    if curve == 'single':
        Calibration = SingleCalibrate()
        if method == 'CG':
            para = Calibration.CG_minimize()
        else:
            para = Calibration.adam_minimize()
    else:
        Calibration = MultipleCalibrate()
        if method == 'CG':
            para = Calibration.CG_minimize()
        else:
            para = Calibration.adam_minimize()
    print(para)

calibrate()