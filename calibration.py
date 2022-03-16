'''
Description: 
Version: 1.0
Author: SHAO Nuoya
Date: 2022-03-16 00:03:06
LastEditors: SHAO Nuoya
LastEditTime: 2022-03-16 16:16:28
'''
import tensorflow as tf
from scipy.optimize import minimize
from single_vasicek import SingleVasicek
from multiple_vasicek import MultipleVasicek
import numpy as np


class SingleCalibrate:
    def __init__(self) -> None:
        pass

    def marginal_liklihood(self, para):
        r0, k, theta, sigma = para
        VS = SingleVasicek(r0, k, theta, sigma)
        ts = np.linspace(0, 1, VS.T)

        rts = VS.generate_rt(ts)

        y = VS.Log_Pt(rts)
        mu_y = VS.mu(ts)

        Sigma_yy = VS.Sigma(ts, ts, np.var(y))

        res = -(y - mu_y).T @ np.linalg.inv(Sigma_yy) @ (y - mu_y) / 2
        return res.ravel()

    def CG_minimize(self):
        para = [0.1, 0.1, 0.1, 0.1]
        res = minimize(self.marginal_liklihood,
                       para,
                       method='CG',
                       options={'disp': True})

    def adam_minimize(self):
        opt = tf.keras.optimizers.Adam(learning_rate=0.1)
        r0 = tf.Variable(0.1)
        k = tf.Variable(0.1)
        theta = tf.Variable(0.1)
        sigma = tf.Variable(0.1)

        para = [r0, k, theta, sigma]
        step_count = opt.minimize(self.marginal_liklihood(para), para).numpy()

        return [r0.numpy(), k.numpy(), theta.numpy(), sigma.numpy()]


#*************************Multiple curve calibration***************************
class MultipleCalibrate:
    def __init__(self) -> None:
        pass
    
    def marginal_liklihood(self, para):
        r01, k1, theta1, sigma1, r02, k2, theta2, sigma2 = para
        VS = MultipleVasicek(r01, k1, theta1, sigma1, r02, k2, theta2, sigma2)

        ts = np.linspace(0, 1, VS.T)
        rts = VS.generate_rt(ts)
        rts0 = rts[:, 0]

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

    def CG_minimize(self):
        para0 = [0.1 for _ in range(8)]
        res = minimize(
            self.marginal_liklihood,
            para0,
            method='CG',
            #jac=self.jac_marginal_liklihood,
            options={'disp': True})

    def adam_minimize(self):
        opt = tf.keras.optimizers.Adam(learning_rate=0.1)
        r01 = tf.Variable(0.1)
        k1 = tf.Variable(0.1)
        theta1 = tf.Variable(0.1)
        sigma1 = tf.Variable(0.1)
        r02 = tf.Variable(0.1)
        k2 = tf.Variable(0.1)
        theta2 = tf.Variable(0.1)
        sigma2 = tf.Variable(0.1)

        para = [r01, k1, theta1, sigma1, r02, k2, theta2, sigma2]
        step_count = opt.minimize(lambda: self.marginal_liklihood(para), para)

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
    pass