'''
Description: 
Version: 1.0
Author: SHAO Nuoya
Date: 2022-03-16 00:03:06
LastEditors: SHAO Nuoya
LastEditTime: 2022-04-24 02:44:42
'''
import tensorflow as tf
from scipy.optimize import minimize, shgo
from single_vasicek import SingleVasicek
from multiple_vasicek import MultipleVasicek
import numpy as np


class SingleCalibrate:
    def __init__(self) -> None:
        simulated_VS = SingleVasicek(r0=0.5, k=2, theta=0.1, sigma=0.2)
        self.ts = np.linspace(0, 1, simulated_VS.T + 1)[:-1]
        rs = simulated_VS.generate_rt(self.ts)
        self.y = simulated_VS.Log_Pt(rs[1:])
        self.iter = 0

    def minus_marginal_liklihood(self, para):
        self.iter += 1
        r0, k, theta, sigma = para
        VS = SingleVasicek(r0, k, theta, sigma)

        mu_y = VS.mu(self.ts[1:])
        SigmaYY = VS.Sigma(self.ts[1:], self.ts[1:])

        # rs = VS.generate_rt(self.ts)
        # y = VS.Log_Pt(rs)
        # res = -(y - mu_y).T @ np.linalg.inv(SigmaYY) @ (y - mu_y) / 2
        res = -(self.y - mu_y).T @ np.linalg.inv(SigmaYY) @ (self.y - mu_y) / 2
        res = -res[0][0]
        print(
            f"{self.iter}\t{para[0]:.8f}\t{para[1]:.8f}\t{para[2]:.8f}\t{para[3]:.8f}\t{res:.8f}"
        )
        return res

    def Global_minimize(self):
        bounds = [(0, 1), (1, 3), (0, 0.5), (0, 0.5)]
        epsilon = 0.1
        cons = (
            {
                'type': 'ineq',
                'fun': lambda x: x[0] - epsilon
            },
            {
                'type': 'ineq',
                'fun': lambda x: x[1] - epsilon
            },
            {
                'type': 'ineq',
                'fun': lambda x: x[2] - epsilon
            },
            {
                'type': 'ineq',
                'fun': lambda x: x[3] - epsilon
            },
        )
        res = shgo(self.minus_marginal_liklihood,
                   bounds=bounds,
                   options={'disp': True},
                   n=2**6,
                   constraints=cons,
                   sampling_method='sobol')
        return res

    def CG_minimize(self):
        para = [0.5 for _ in range(4)]
        res = minimize(self.minus_marginal_liklihood,
                       para,
                       method='CG',
                       options={'disp': True})
        return res

    def adam_minimize(self):
        opt = tf.keras.optimizers.Adam(learning_rate=0.1)
        r0 = tf.Variable(0.1)
        k = tf.Variable(0.1)
        theta = tf.Variable(0.1)
        sigma = tf.Variable(0.1)

        para = [r0, k, theta, sigma]
        step_count = opt.minimize(lambda: self.minus_marginal_liklihood(para),
                                  para).numpy()

        return [r0.numpy(), k.numpy(), theta.numpy(), sigma.numpy()]


#*************************Multiple curve calibration***************************
class MultipleCalibrate:
    def __init__(self) -> None:
        simulated_VS = MultipleVasicek(r01=0.5,
                                       k1=2,
                                       theta1=0.1,
                                       sigma1=0.2,
                                       r02=0.7,
                                       k2=0.5,
                                       theta2=0.03,
                                       sigma2=0.8)
        ts = np.linspace(0, 1, simulated_VS.T)[1:-1]
        rts = simulated_VS.generate_rt(ts)
        rts0 = rts[:, 0]

        self.y_delta = simulated_VS.Log_Pt_delta(rts)
        self.y0 = simulated_VS.Log_Pt(rts0)
        self.iter = 0

    def minus_marginal_liklihood(self, para):
        self.iter += 1

        r01, k1, theta1, sigma1, r02, k2, theta2, sigma2 = para
        VS = MultipleVasicek(r01, k1, theta1, sigma1, r02, k2, theta2, sigma2)

        ts = np.linspace(0, 1, VS.T)[1:-1]
        y = np.vstack((self.y0, self.y_delta))
        mu_y = np.vstack((VS.mu0(ts), VS.mu_delta(ts)))

        Sigma_y00 = VS.Sigma(ts, ts, np.var(self.y0))
        Sigma_ydd = VS.Sigma(ts, ts, np.var(self.y_delta), delta=True)
        Sigma_y0d = VS.Sigma0_delta(ts)
        Sigma_yd0 = VS.Sigmadelta_0(ts)
        Sigma_y_upper = np.hstack((Sigma_y00, Sigma_y0d))
        Sigma_y_lower = np.hstack((Sigma_yd0, Sigma_ydd))
        Sigma_y = np.vstack((Sigma_y_upper, Sigma_y_lower))

        res = -(y - mu_y).T @ np.linalg.inv(Sigma_y) @ (y - mu_y) / 2
        res = -res[0][0]
        print(
            f"{self.iter}\t{para[0]:.8f}\t{para[1]:.8f}\t{para[2]:.8f}\t{para[3]:.8f}\t{res:.8f}"
        )
        return res

    def CG_minimize(self):
        para0 = [0.1 for _ in range(8)]
        res = minimize(
            self.minus_marginal_liklihood,
            para0,
            method='CG',
            #jac=self.jac_marginal_liklihood,
            options={'disp': True})
        return res

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
        step_count = opt.minimize(lambda: self.minus_marginal_liklihood(para),
                                  para)

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