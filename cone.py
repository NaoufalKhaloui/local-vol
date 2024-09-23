
import numpy as np
from scipy.stats import norm

cutoff = .99
deltas = np.linspace(1.-cutoff, cutoff, num=19)
f2s = norm.ppf(deltas)

class interp_one:
    def __init__(self, spline, spline1d, left, right, spline2d=None):
        self.spline=spline
        self.spline1d=spline1d
        self.spline2d=spline2d
        self.left = left
        self.right = right
    def __call__(self, x):
        out = self.spline(x)
        if x[0]<=self.left[0]:
            for ii in range(len(x)):
                if x[ii]<=self.left[0]:
                    out[ii]=self.left[1]
                else:
                    break
        if x[-1]>=self.right[0]:
            for ii in range(len(x)-1,0,-1):
                if x[ii]>=self.right[0]:
                    out[ii]=self.right[1]
                else:
                    break
        return out

    def derivative(self, x):
        out = self.spline1d(x)
        if x[0]<=self.left[0]:
            for ii in range(len(x)):
                if x[ii]<=self.left[0]:
                    out[ii]=0.
                else:
                    break
        if x[-1]>=self.right[0]:
            for ii in range(len(x)-1,0,-1):
                if x[ii]>=self.right[0]:
                    out[ii]=0.
                else:
                    break
        return out

    def dderivative(self, x):
        out = self.spline2d(x)
        if x[0]<=self.left[0]:
            for ii in range(len(x)):
                if x[ii]<=self.left[0]:
                    out[ii]=0.
                else:
                    break
        if x[-1]>=self.right[0]:
            for ii in range(len(x)-1,0,-1):
                if x[ii]>=self.right[0]:
                    out[ii]=0.
                else:
                    break
        return out



def portfolio_f2s_from_strikes(strikes, total_vols):
    return strikes/total_vols + .5 * total_vols

def portfolio_f1s_from_strikes(strikes, total_vols):
    return strikes/total_vols - .5 * total_vols

def implied_volatility_at_delta_log(interpolator, ttx, f2s = f2s, max_vol_down=2,  max_vol_up=2, n_points=200, tolerance=1e-7, max_iter=20):
    sqrt_ttx = np.sqrt(ttx)
    max_total_vol_down = max_vol_down * sqrt_ttx
    max_total_vol_up = max_vol_up * sqrt_ttx
    x_L = (f2s[0] - .5 * max_total_vol_down) * max_total_vol_down
    x_H = (f2s[-1] + .5 * max_total_vol_up) * max_total_vol_up
    x_strikes = np.linspace(x_L, x_H, n_points)
    temp_total_vols = interpolator(x_strikes) * sqrt_ttx
    temp_f2s = x_strikes / temp_total_vols + .5 * temp_total_vols
    L_check = np.searchsorted(f2s, temp_f2s, side='left')
    R_check = np.searchsorted(f2s, temp_f2s, side='right')
    if (L_check[0] >= len(f2s)) or (0 == R_check[-1]):
        print('GRID SIZING PROBLEM at', ttx)

    f2s = f2s[L_check[0]:R_check[-1]]


    LL = np.searchsorted(temp_f2s, f2s, side='left')
    lin_interp = (f2s - temp_f2s[LL - 1]) / (temp_f2s[LL] - temp_f2s[LL - 1])
    strikes_interp = x_strikes[LL - 1] + (x_strikes[LL] - x_strikes[LL - 1]) * lin_interp
    out_temp = np.NaN
    for hh in range(max_iter) :
        out_temp = interpolator(strikes_interp) * sqrt_ttx
        current_f2s = portfolio_f2s_from_strikes(strikes_interp, out_temp)
        if not np.all(np.diff(current_f2s) > 0.0001):
            print('Monotonicity ARB at ', ttx)
        abs_error = np.sum(np.abs(current_f2s - f2s))
        if abs_error < tolerance:
            break
        deriv_inv = out_temp / (1. - interpolator.derivative(strikes_interp) * sqrt_ttx * portfolio_f1s_from_strikes(strikes_interp, out_temp))

        strikes_interp -= (current_f2s - f2s) * deriv_inv

    if hh == (max_iter-1):
        print("DID NOT CONVERGE")
    return out_temp / sqrt_ttx, strikes_interp


def linear_interpolator(x, input_matrix, tenors):
    time = np.array([0.]+tenors.tolist())

    input_matrix_ = np.zeros((len(time), input_matrix.shape[1]))
    input_matrix_[1:,:] = input_matrix

    LL = np.searchsorted(time, x, side = 'left')
    lin_interp = (x-time[LL-1])/(time[LL]-time[LL-1])
    output_matrix = input_matrix_[LL-1,:]+(input_matrix_[LL,:]-input_matrix_[LL-1])*lin_interp[:,np.newaxis]
    return output_matrix

def linear_slice_interpolator(x, input_y, input_x):
    x = np.maximum(x, input_x[0])
    x = np.minimum(x, input_x[-1])


    LL = np.searchsorted(input_x, x, side = 'left')
    lin_interp = (x-input_x[LL-1])/(input_x[LL]-input_x[LL-1])
    output_matrix = input_y[LL-1]+(input_y[LL]-input_y[LL-1])*lin_interp
    return output_matrix