'''
Description: 
Version: 1.0
Author: SHAO Nuoya
Date: 2022-03-16 00:19:40
LastEditors: SHAO Nuoya
LastEditTime: 2022-04-11 01:08:18
'''
from single_vasicek import SingleVasicek
from multiple_vasicek import MultipleVasicek
from calibration import SingleCalibrate, MultipleCalibrate
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time


#***********************Prediction**************************
def single_curve_prediction(CI=0.95):
    VS = SingleVasicek()

    ts = np.linspace(0, 1, VS.T)[1:-1]
    observation_index = range(0, len(ts) // 2, 20)
    prediction_index = range(len(ts) // 2, len(ts), 20)

    ts_observation = ts[observation_index]
    ts_prediction = ts[prediction_index]

    rs = VS.generate_rt(ts)  # daily rate array

    # daily log bond price array
    log_Pt = VS.Log_Pt(rs)
    log_Pt_observation = log_Pt[observation_index]

    MuY = VS.mu(ts_observation)  # expectation of log bond price (observations)
    MuY_ = VS.mu(ts_prediction)  # expectation of log bond price (predictions)

    SigmaY_Y = VS.Sigma(ts_prediction, ts_observation)
    SigmaYY_ = VS.Sigma(ts_observation, ts_prediction)
    SigmaY_Y_ = VS.Sigma(ts_prediction, ts_prediction)
    SigmaYY = VS.Sigma(ts_observation, ts_observation)

    # conditional expectation of log bond price (predictions)
    mu_ = MuY_ + SigmaY_Y @ np.linalg.inv(SigmaYY) @ (log_Pt_observation - MuY)
    Sigma_ = SigmaY_Y_ - SigmaY_Y @ np.linalg.inv(SigmaYY) @ SigmaYY_

    plt.plot(ts, log_Pt, label='log bond price')
    plt.scatter(ts_observation,
                log_Pt_observation,
                s=10,
                color='red',
                label='observations')
    plt.plot(ts_prediction, mu_, color='orange', label='predictions')

    var = CI * np.sqrt(np.diagonal(Sigma_).reshape(-1, 1))
    low_CI_bound = list((mu_ - var).flatten())
    high_CI_bound = list((mu_ + var).flatten())

    CI_df = pd.DataFrame.from_dict({
        "CI_x":
        list(ts_prediction) + (list(ts_observation)),
        "CI_y1":
        low_CI_bound + list(log_Pt_observation.flatten()),
        "CI_y2":
        high_CI_bound + list(log_Pt_observation.flatten())
    }).sort_values(by="CI_x")

    plt.fill_between(CI_df["CI_x"].values,
                     CI_df["CI_y1"].values,
                     CI_df["CI_y2"].values,
                     color='gray',
                     alpha=0.25,
                     label='95% CI')
    plt.legend()
    # plt.savefig('Single.png')
    plt.show()


#*************************Calibration***************************
def calibrate(curve='single', method='CG'):
    if curve == 'single':
        Calibration = SingleCalibrate()
        if method == 'CG':
            para = Calibration.CG_minimize()
        elif method == 'adam':
            para = Calibration.adam_minimize()
    elif curve == 'multiple':
        Calibration = MultipleCalibrate()
        if method == 'CG':
            para = Calibration.CG_minimize()
        elif method == 'adam':
            para = Calibration.adam_minimize()
    print(para)


if __name__ == '__main__':
    single_curve_prediction()

    # start = time.time()
    # calibrate(curve='single')
    # end = time.time()
    # print("Time used : ", end - start)

    # # start = time.time()
    # # calibrate(curve='multiple')
    # # end = time.time()
    # # print("Time used : ", end-start)