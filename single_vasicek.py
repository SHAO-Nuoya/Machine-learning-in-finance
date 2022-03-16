'''
Description: 
Version: 1.0
Author: SHAO Nuoya
Date: 2022-03-14 16:11:50
LastEditors: SHAO Nuoya
LastEditTime: 2022-03-16 18:56:44
'''
from numpy import exp
import numpy as np


class SingleVasicek:
    def __init__(self, r0=0.5, k=2, theta=0.1, sigma=0.2, T=252) -> None:
        """
        Args:
            r0 (float, optional): initial interest rate. Defaults to 0.5.
            T (float, optional): maturity. Defaults to 252 days.
        """
        self.r0 = r0
        self.k = k
        self.theta = theta
        self.sigma = sigma
        self.T = T

    def generate_rt(self, ts) -> list[float]:
        """
        Args:
            ts (list[float]): time list (unit is year)
        Returns:
            list: list of interest rate for every day
        """
        k, theta, sigma, r0 = self.k, self.theta, self.sigma, self.r0
        res = [r0]
        dt = (ts[-1] - ts[0]) / len(ts)
        for t in range(1, len(ts)):
            r_pre = res[-1]
            r_new = r_pre + k * (
                theta - r_pre) * dt + sigma * np.random.normal(0, np.sqrt(dt))
            res.append(r_new)

        return res

    def A(self, t):
        k, theta, sigma = self.k, self.theta, self.sigma
        tau = 1 - t
        res = theta / k * (exp(-k * tau) + k * tau - 1) + sigma**2 / (
            4 * k**3) * (exp(-2 * k * tau) - 4 * exp(-k * tau) - 2 * k * tau +
                         3)
        return res

    def B(self, t):
        tau = 1 - t
        res = (1 - exp(-self.k * tau)) / self.k
        return res

    def Log_Pt(self, rts):
        """
        Args:
            rts (list[float]): interest rate list

        Returns:
            float: log bond price
        """
        res = []
        for i, rt in enumerate(rts):
            t = i / len(rts)
            res.append(-self.A(t) - self.B(t) * rt)
        return np.array(res).reshape(-1, 1)

    # todo 是否需要更新rs ?
    def mu_knowing_s(self, t, s):
        pass

    def mu(self, ts):
        """expectation of log P(t, T)"""
        res = []
        for t in ts:
            re = -self.A(t) - self.B(t) * (self.r0 * exp(-self.k * t) +
                                           self.theta * (1 - exp(-self.k * t)))
            res.append(re)
        return np.array(res).reshape(-1, 1)

    def c(self, s, t):
        """covariance function"""
        k, sigma = self.k, self.sigma
        Bs = self.B(s)
        Bt = self.B(t)

        term1 = exp(-k * (s + t))
        term2 = exp(2 * k * min(s, t)) - 1

        return Bs * Bt * sigma**2 / (2 * k) * term1 * term2

    def Sigma(self, Ss, Ts, sigma_hat=0):
        """
        Args:
            Ss (_type_): time list1
            Ts (_type_): time list2
            sigma_hat (int, optional): empirical variance. Defaults to 0.

        Returns:
            np.array: 
        """
        res = np.zeros((len(Ss), len(Ts)))
        for i, s in enumerate(Ss):
            for j, t in enumerate(Ts):
                res[i, j] = self.c(s, t) + sigma_hat * (i == j)

        return res


if __name__ == "__main__":
    pass
