'''
Description: 
Version: 1.0
Author: SHAO Nuoya
Date: 2022-03-14 16:11:50
LastEditors: SHAO Nuoya
LastEditTime: 2022-03-14 21:14:34
'''
from numpy import exp
import numpy as np


class SingleVasicek:
    def __init__(self, r0=0.5, k=2, theta=0.1, sigma=0.2, T=252) -> None:
        """_summary_
        Args:
            r0 (float, optional): initial interest rate. Defaults to 0.5.
            T (float, optional): maturity. Defaults to 1.
            N (int, optional): time step. Defaults to 200.
        """
        self.r0 = r0
        self.k = k
        self.theta = theta
        self.sigma = sigma
        self.T = T

    def generate_rt(self, ts):
        k, theta, sigma, r0 = self.k, self.theta, self.sigma, self.r0
        res = [r0]
        dt = (ts[-1] - ts[0]) / len(ts)
        for t in range(1, len(ts)):
            r_pre = res[-1]
            r_new = r_pre + k * (
                theta - r_pre) * dt + sigma * np.random.normal(0, np.sqrt(dt))
            res.append(r_new)

        return res

    def Log_Pt(self, t, rt):
        k, theta, sigma = self.k, self.theta, self.sigma
        tau = 1 - t
        A = theta / k * (exp(-k * tau) + k * tau - 1) + sigma**2 / (
            4 * k**3) * (exp(-2 * k * tau) - 4 * exp(-k * tau) - 2 * k * tau +
                         3)
        B = (1 - exp(-k * tau)) / k
        return -A - B * rt

    def mu_knowing_s(self, t, s):
        pass

    def mu(self, t):
        """expectation of log P(t, T)"""
        k, theta, sigma, r0 = self.k, self.theta, self.sigma, self.r0
        tau = 1 - t
        A = theta * (exp(-k * tau) + k * tau - 1) / k + sigma**2 * (exp(
            -2 * k * tau) - 4 * exp(-k * tau) - 2 * k * tau + 3) / (4 * k**3)
        B = (1 - exp(-k * tau)) / k
        return -A - B * (r0 * exp(-k * t) + theta * (1 - exp(-k * t)))

    def c(self, s, t):
        """covariance function"""
        k, sigma = self.k, self.sigma
        Bs = (1 - exp(-k * (1 - s))) / k
        Bt = (1 - exp(-k * (1 - t))) / k

        a = exp(-k * (s + t))
        b = (exp(2 * k * min(s, t)) - 1)

        return Bs * Bt * sigma**2 / (2 * k) * exp(
            -k * (s + t)) * (exp(2 * k * min(s, t)) - 1)

    def Sigma(self, sigma_hat):
        res = np.zeros((self.T, self.T))
        for i in self.T:
            for j in self.T:
                res[i, j] = self.c(i, j) + sigma_hat**2 * (i == j)
        return res


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    VS = SingleVasicek()

    ts = np.linspace(0, 1, VS.T)
    ts_test = np.linspace(0, 1, VS.T // 2)

    SigmaY_Y = np.zeros((len(ts_test), len(ts)))
    SigmaYY_ = np.zeros((len(ts), len(ts_test)))
    SigmaY_Y_ = np.zeros((len(ts_test), len(ts_test)))
    SigmaYY = np.zeros((len(ts), len(ts)))

    rts = VS.generate_rt(ts)
    y = np.array([VS.Log_Pt(i / len(rts), rt)
                  for i, rt in enumerate(rts)]).reshape(-1, 1)

    MuY_ = np.array([VS.mu(s) for s in ts_test]).reshape(-1, 1)
    MuY = np.array([VS.mu(t) for t in ts]).reshape(-1, 1)

    for i, s in enumerate(ts_test):
        for j, t in enumerate(ts):
            SigmaY_Y[i, j] = VS.c(s, t)
            SigmaYY_[j, i] = VS.c(t, s)

    for i, si in enumerate(ts_test):
        for j, sj in enumerate(ts_test):
            SigmaY_Y_[i, j] = VS.c(si, sj)

    for i, ti in enumerate(ts):
        for j, tj in enumerate(ts):
            SigmaYY[i, j] = VS.c(ti, tj) + np.var(y) * (i == j)

    mu_ = MuY_ + SigmaY_Y @ np.linalg.inv(SigmaYY) @ (y - MuY)

    plt.scatter(ts, y, s=20, label='observation')
    plt.plot(ts_test, mu_, color='red', label='prediction')
    plt.legend()
    plt.savefig('Single.png')
    plt.show()
