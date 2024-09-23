import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from scipy.stats import norm
from dataclasses import dataclass
from typing import Union, Tuple
from enum import Enum

class BetaInterpolation(Enum):
    LINEAR = 1
    UNKNOWN = 2


@dataclass
class BlendingScheme:
    start: float
    end: float
    beta_interpolation: BetaInterpolation = BetaInterpolation.LINEAR

    def blending_beta(self, t):
        if t <= self.start:
            return 1.0
        elif self.start < t <= self.end:
            return 0.5
        else:
            return 0.0

    def calc_betas(self, exp_div_dates: np.ndarray):
        betas = np.array([self.blending_beta(t_k) for t_k in exp_div_dates])
        return betas


def get_next_div_date(t, exp_div_dates, T) -> float:
    if exp_div_dates.size == 0 or exp_div_dates[0]>T:
        return T
    idx = np.searchsorted(exp_div_dates, t, side='right').astype(int)
    if idx > exp_div_dates.size - 1 or   exp_div_dates[idx] > T:
        return T
    else:
        return exp_div_dates[idx]


class ForwardCalculator:
    def __init__(self, blending_scheme: BlendingScheme,
                 s0: float,
                 r: float,
                 q: float,
                 exp_div_dates: np.ndarray,
                 exp_divs: np.ndarray):
        self.s0 = s0
        self.r = r
        self.q = q
        self.blending_scheme = blending_scheme
        self.exp_div_dates = exp_div_dates
        self.exp_divs = exp_divs

    def __calc_forward_and_cashdiv_sum(self,
                                       t: float,
                                       T: float) -> Tuple[float, float]:
        s0 = self.s0
        r = self.r
        q = self.q
        exp_div_dates = self.exp_div_dates
        exp_divs = self.exp_divs

        betas = self.blending_scheme.calc_betas(exp_div_dates)
        if np.isposinf(T):
            T = exp_div_dates[-1] 

        assert not (np.isinf(T) and np.isnan(T))
        t1 = get_next_div_date(t, exp_div_dates, T)
        growth_fact = np.exp((t1 - t) * (r - q))
        cashdiv_sum = 0.0
        div_idxs = np.where((exp_div_dates > t) & (exp_div_dates <= T))[0]

        b = np.zeros_like(exp_div_dates)
        a = np.zeros_like(exp_div_dates)
        for k in div_idxs:
            t_k = exp_div_dates[k]
            delta_k = exp_divs[k]
            beta_k = betas[k]
            b[k] = beta_k * delta_k

            fwd = (s0 - cashdiv_sum) * growth_fact
            a[k] = delta_k * (1.0 - beta_k) / fwd

            t_next = get_next_div_date(t_k, exp_div_dates, T)

            # update
            cashdiv_sum = cashdiv_sum + b[k] / (growth_fact*(1-a[k]))
            growth_fact = growth_fact * (1 - a[k]) * np.exp((t_next - t_k) * (r - q))


        fwd = (s0 - cashdiv_sum) * growth_fact
        return fwd, cashdiv_sum, a, b