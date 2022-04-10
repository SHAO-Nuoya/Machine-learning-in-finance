'''
Description: 
Version: 1.0
Author: SHAO Nuoya
Date: 2022-03-14 21:18:14
LastEditors: SHAO Nuoya
LastEditTime: 2022-03-17 01:15:53
'''

from numpy import exp, sqrt
import numpy as np


class MultipleVasicek:
    def __init__(self,
                 r01=0.5,
                 k1=2,
                 theta1=0.1,
                 sigma1=0.2,
                 r02=0.7,
                 k2=0.5,
                 theta2=0.03,
                 sigma2=0.8,
                 rho=0.2,
                 T=127) -> None:
        """
        Args:
            T (float, optional): maturity. Defaults to 127 days.
        """
        self.r01 = r01
        self.k1 = k1
        self.theta1 = theta1
        self.sigma1 = sigma1
        self.r02 = r02
        self.k2 = k2
        self.theta2 = theta2
        self.sigma2 = sigma2
        self.rho = rho
        self.T = T

    def generate_rt(self, ts) -> np.ndarray:
        """
        Args:
            ts (list[float]): time list (unit is year)
        Returns:
            list: list of interest rate for every day
        """
        k1, theta1, sigma1, r01 = self.k1, self.theta1, self.sigma1, self.r01
        k2, theta2, sigma2, r02 = self.k2, self.theta2, self.sigma2, self.r02
        rho = self.rho
        res = np.zeros((len(ts), 2))
        res[0, 0] = r01
        res[0, 1] = r02
        dt = (ts[-1] - ts[0]) / len(ts)
        for t in range(1, len(ts)):
            dw1, dw2 = np.random.normal(0, 1, 2)
            dw_1 = dw1
            dw_2 = rho * dw1 + sqrt(1 - rho**2) * dw2

            r_pre1 = res[t - 1, 0]
            r_new1 = r_pre1 + k1 * (theta1 -
                                    r_pre1) * dt + sigma1 * sqrt(dt) * dw_1

            r_pre2 = res[t - 1, 1]
            r_new2 = r_pre2 + k2 * (theta2 -
                                    r_pre2) * dt + sigma2 * sqrt(dt) * dw_2
            res[t, 0] = r_new1
            res[t, 1] = r_new2
        return res

    def A(self, t):
        k1, theta1, sigma1 = self.k1, self.theta1, self.sigma1
        tau = 1 - t
        res = theta1 / k1 * (exp(-k1 * tau) + k1 * tau - 1) + sigma1**2 / (
            4 * k1**3) * (exp(-2 * k1 * tau) - 4 * exp(-k1 * tau) -
                          2 * k1 * tau + 3)
        return res

    def B(self, t):
        tau = 1 - t
        res = (1 - exp(-self.k1 * tau)) / self.k1
        return res

    def Psi1(self, t):
        return -self.B(t)

    def Psi2(self, t):
        tau = 1 - t
        res = (1 - exp(-self.k2 * tau)) / self.k2
        return res

    def Phi(self, t):
        k1, theta1, sigma1 = self.k1, self.theta1, self.sigma1
        k2, theta2, sigma2 = self.k1, self.theta1, self.sigma1
        rho = self.rho
        tau = 1 - t

        term1 = -(theta1 - theta2) * tau - theta1 / k1 * (
            exp(-k1 * tau) - 1) + theta2 / k2 * (exp(-k2 * tau) - 1)
        term2 = sigma1**2 / (2 * k1**2) * (tau + 2 / k1 * exp(-k1 * tau) - 1 /
                                           (2 * k1) * exp(-2 * k1 * tau) - 3 /
                                           (2 * k1))
        term3 = sigma2**2 / (2 * k2**2) * (tau + 2 / k2 * exp(-k2 * tau) - 1 /
                                           (2 * k2) * exp(-2 * k2 * tau) - 3 /
                                           (2 * k2))
        term4 = tau + (exp(-k1 * tau) - 1) / k1 + (exp(-k2 * tau) - 1) / k2 - (
            exp(-(k1 + k2) * tau) - 1) / (k1 + k2)

        return term1 + term2 + term3 - rho * sigma1 * sigma2 / k1 / k2 * term4

    def mu0(self, ts):
        """expectation of log P(t, T)"""
        k1, theta1, r01 = self.k1, self.theta1, self.r01
        res = []
        for t in ts:
            re = -self.A(t) - self.B(t) * (r01 * exp(-k1 * t) + theta1 *
                                           (1 - exp(-k1 * t)))
            res.append(re)
        return np.array(res).reshape(-1, 1)

    def c0(self, s, t):
        """covariance function"""
        k1, sigma1 = self.k1, self.sigma1
        Bs = self.B(s)
        Bt = self.B(t)

        term1 = exp(-k1 * (s + t))
        term2 = (exp(2 * k1 * min(s, t)) - 1)

        return Bs * Bt * sigma1**2 / (2 * k1) * term1 * term2

    def mu_delta(self, ts):
        res = []
        for t in ts:
            term1 = self.Psi1(t) * (self.r01 * exp(-self.k1 * t) +
                                    self.theta1 * (1 - exp(-self.k1 * t)))
            term2 = self.Psi2(t) * (self.r02 * exp(-self.k2 * t) +
                                    self.theta2 * (1 - exp(-self.k2 * t)))
            res.append(self.Phi(t) + term1 + term2)
        return np.array(res).reshape(-1, 1)

    def c_delta(self, s, t):
        """covariance function"""
        k1, sigma1 = self.k1, self.sigma1
        k2, sigma2 = self.k1, self.sigma1
        rho = self.rho

        term1 = self.Psi1(s) * self.Psi1(t) * sigma1**2 / 2 / k1 * exp(
            -k1 * (s + t)) * (exp(2 * k1 * min(s, t)) - 1)
        term2 = self.Psi1(s) * self.Psi2(t) * rho * sigma1 * sigma2 / (
            k1 + k2) * exp(-k1 * s - k2 * t) * (exp((k1 + k2) * min(s, t)) - 1)
        term3 = self.Psi1(t) * self.Psi2(s) * rho * sigma1 * sigma2 / (
            k1 + k2) * exp(-k1 * t - k2 * s) * (exp((k1 + k2) * min(s, t)) - 1)
        term4 = self.Psi2(s) * self.Psi2(t) * sigma2**2 / 2 / k2 * exp(
            -k2 * (s + t)) * (exp(2 * k2 * min(s, t)) - 1)
        return term1 + term2 + term3 + term4

    def Sigma(self, Ss, Ts, sigma_hat=0, delta=False):
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
                if delta:
                    res[i, j] = self.c_delta(s, t) + sigma_hat * (i == j)
                else:
                    res[i, j] = self.c0(s, t) + sigma_hat * (i == j)
        return res

    def Sigma0_delta(self, Ts):
        k1, sigma1 = self.k1, self.sigma1
        k2, sigma2 = self.k1, self.sigma1
        rho = self.rho
        res = np.zeros((len(Ts), len(Ts)))
        for i, ti in enumerate(Ts):
            for j, tj in enumerate(Ts):
                term1 = self.Psi1(ti) * self.Psi1(
                    tj) * sigma1**2 / 2 / k1 * exp(
                        -k1 * (ti + tj)) * (exp(2 * k1 * min(ti, tj)) - 1)
                term2 = self.Psi1(ti) * self.Psi2(
                    tj) * rho * sigma1 * sigma2 / (
                        k1 + k2) * exp(-k1 * ti - k2 * tj) * (exp(
                            (k1 + k2) * min(ti, tj)) - 1)
                res[i, j] = term1 + term2
        return res

    def Sigmadelta_0(self, Ts):
        k1, sigma1 = self.k1, self.sigma1
        k2, sigma2 = self.k1, self.sigma1
        rho = self.rho
        res = np.zeros((len(Ts), len(Ts)))
        for i, ti in enumerate(Ts):
            for j, tj in enumerate(Ts):
                term1 = self.Psi1(ti) * self.Psi1(
                    tj) * sigma1**2 / 2 / k1 * exp(
                        -k1 * (ti + tj)) * (exp(2 * k1 * min(ti, tj)) - 1)
                term2 = self.Psi1(tj) * self.Psi2(
                    ti) * rho * sigma1 * sigma2 / (
                        k1 + k2) * exp(-k1 * tj - k2 * ti) * (exp(
                            (k1 + k2) * min(ti, tj)) - 1)
                res[i, j] = term1 + term2
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

    def Log_Pt_delta(self, rts):
        """
        Args:
            rts (np.ndarray[float]): interest rate with dimension 2: array(n, 2)

        Returns:
            float: log bond price
        """
        res = []
        for i, rt in enumerate(rts):
            t = i / len(rts)
            psi = np.array([self.Psi1(t), self.Psi2(t)]).reshape(1, -1)
            res.append(self.Phi(t) + psi @ rt)
        return np.array(res).reshape(-1, 1)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    VS = MultipleVasicek()

    ts = np.linspace(0, 1, VS.T)[1:-1]

    rts = VS.generate_rt(ts)
    rts1 = rts[:, 0]

    y_delta = VS.Log_Pt_delta(rts)
    y = VS.Log_Pt(rts1)
    plt.plot(rts[:, 0], label='0')
    plt.plot(rts[:, 1], label='1')
    plt.plot(y_delta, label='delta')
    plt.plot(y, label='y')
    plt.legend()
    plt.show()
