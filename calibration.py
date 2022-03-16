'''
Description: 
Version: 1.0
Author: SHAO Nuoya
Date: 2022-03-16 00:03:06
LastEditors: SHAO Nuoya
LastEditTime: 2022-03-16 01:50:37
'''
import tensorflow as tf
from scipy.optimize import minimize
from single_vasicek import SingleVasicek
from multiple_vasicek import MultipleVasicek
import numpy as np


class single_calibrate:
    def __init__(self) -> None:
        pass

    def marginal_liklihood(self, r0, k, theta, sigma):
        pass

    def jac_marginal_liklihood(self, r0, k, theta, sigma):
        pass

    def CG_minimize(self):
        para0 = [0.0, 0.0, 0.0, 0.0]
        res = minimize(self.marginal_liklihood,
                       para0,
                       method='CG',
                       jac=self.jac_marginal_liklihood,
                       options={'disp': True})

    def adam_minimize(self):
        opt = tf.keras.optimizers.Adam(learning_rate=0.1)
        r0 = tf.Variable(0.0)
        k = tf.Variable(0.0)
        theta = tf.Variable(0.0)
        sigma = tf.Variable(0.0)

        step_count = opt.minimize(self.marginal_liklihood,
                                  [r0, k, theta, sigma]).numpy()

        return [r0.numpy(), k.numpy(), theta.numpy(), sigma.numpy()]


#*************************Multiple curve calibration*************************8*
def multi_marginal_liklihood(para):
    r01, k1, theta1, sigma1, r02, k2, theta2, sigma2 = para
    VS = MultipleVasicek()
    ts = np.linspace(0, 1, VS.T)

    rts = VS.generate_rt(ts)
    rts0 = rts[:, 0]

    VS = MultipleVasicek(r01, k1, theta1, sigma1, r02, k2, theta2, sigma2)
    y_delta = VS.Log_Pt_delta(rts)
    y0 = VS.Log_Pt(rts0)

    y = np.vstack((y0, y_delta))
    mu_y = np.vstack((VS.mu0(ts), VS.mu_delta(ts)))

    Sigma_y00 = VS.Sigma(ts, ts, np.var(y0))
    Sigma_ydd = VS.Sigma(ts, ts, np.var(y_delta), delta=True)
    Sigma_y0d = VS.Sigma0_delta(ts)
    Sigma_yd0 = VS.Sigmadelta_0(ts)
    Sigma_y_upper = np.hstack((Sigma_y00, Sigma_y0d))
    Sigma_y_lower = np.hstack((Sigma_yd0, Sigma_ydd))
    Sigma_y = np.vstack((Sigma_y_upper, Sigma_y_lower))

    res = -(y - mu_y).T @ np.linalg.inv(Sigma_y) @ (y - mu_y) / 2
    return res.ravel()

class multiple_calibrate:
    def __init__(self) -> None:
        pass

    def CG_minimize(self):
        para0 = [0 for _ in range(8)]
        res = minimize(
            multi_marginal_liklihood,
            para0,
            method='CG',
            #jac=self.jac_marginal_liklihood,
            options={'disp': True})

    def adam_minimize(self):
        opt = tf.keras.optimizers.Adam(learning_rate=0.1)
        r01 = tf.Variable(0.0)
        k1 = tf.Variable(0.0)
        theta1 = tf.Variable(0.0)
        sigma1 = tf.Variable(0.0)
        r02 = tf.Variable(0.0)
        k2 = tf.Variable(0.0)
        theta2 = tf.Variable(0.0)
        sigma2 = tf.Variable(0.0)
        
        para = [r01, k1, theta1, sigma1, r02, k2, theta2, sigma2]
        step_count = opt.minimize(lambda:multi_marginal_liklihood(para), para).numpy()

        print(step_count)
        return [
            r01.numpy(),
            k1.numpy(),
            theta1.numpy(),
            sigma1.numpy(),
            r02.numpy(),
            k2.numpy(),
            theta2.numpy(),
            sigma2.numpy()
        ]


if __name__ == "__main__":
    Cali = multiple_calibrate()
    para = Cali.adam_minimize()
    print(para)