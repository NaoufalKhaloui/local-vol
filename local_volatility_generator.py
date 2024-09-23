
import numpy as np
from scipy.stats import norm
import scipy.interpolate as sci

from cone import implied_volatility_at_delta_log, linear_interpolator, interp_one

def local_volatility_grid(T_, strikes_, vol_grid_, fwds_, timesteps, local_vol_x_grid_=None, cutoff_=0.99, num_x_spacepoints_=200):


    spline_list = []
    d1_spline_list = []
    d2_spline_list = []
    interps_list = []

    if local_vol_x_grid_ is None:

        log_moneyness_ = np.log(strikes_[:, np.newaxis] / fwds_[np.newaxis, :])

        for ii in range(len(T_)):
            spline_list.append(sci.CubicSpline(log_moneyness_[:, ii], vol_grid_[ii]))
            d1_spline_list.append(spline_list[ii].derivative(1))
            d2_spline_list.append(spline_list[ii].derivative(2))
            interps_list.append(interp_one(spline_list[ii], d1_spline_list[ii], [log_moneyness_[:, ii][0], vol_grid_[ii][0]], [log_moneyness_[:, ii][-1], vol_grid_[ii][-1]], d2_spline_list[ii]))

        deltas = np.linspace(1. - cutoff_, cutoff_, num=19)

        f2s = norm.ppf(deltas)

        end_expiry_index = len(fwds_) - 1

        end_forward = fwds_[end_expiry_index]

        delta_strike_matrix = np.empty((len(T_), len(f2s)))
        strike_vol_matrix = np.empty((len(T_), num_x_spacepoints_))

        for ii in range(len(T_)):
            see = implied_volatility_at_delta_log(interps_list[ii], T_[ii], f2s)
            delta_strike_matrix[ii,:] = see[1]

        end_x_slice = [delta_strike_matrix[end_expiry_index, 0], delta_strike_matrix[end_expiry_index, -1]]

        local_vol_x_grid_ = np.linspace(end_x_slice[0], end_x_slice[1], num_x_spacepoints_)

        for ii in range(len(T_)):
            strike_vol_matrix[ii, :] = interps_list[ii](local_vol_x_grid_)
    else:
        num_x_spacepoints_ = len(local_vol_x_grid_)
        strike_vol_matrix = vol_grid_

        for ii in range(len(T_)):

            spline_list.append(sci.CubicSpline(local_vol_x_grid_, strike_vol_matrix[ii]))
            d1_spline_list.append(spline_list[ii].derivative(1))
            d2_spline_list.append(spline_list[ii].derivative(2))
            interps_list.append(interp_one(spline_list[ii], d1_spline_list[ii], [local_vol_x_grid_[0], vol_grid_[ii][0]],
            [local_vol_x_grid_[-1], strike_vol_matrix[ii][-1]], d2_spline_list[ii]))

    strike_var_deriv_matrix = np.empty((len(T_), num_x_spacepoints_))
    strike_var_dderiv_matrix = np.empty((len(T_), num_x_spacepoints_))

    for ii in range(len(T_)):

        temp = interps_list[ii].derivative(local_vol_x_grid_)

        strike_var_deriv_matrix[ii, :] = strike_vol_matrix[ii, :] * temp
        strike_var_dderiv_matrix[ii, :] = temp * temp + strike_vol_matrix[ii, :] * interps_list[ii] .dderivative(local_vol_x_grid_)

    total_var = linear_interpolator(timesteps, strike_vol_matrix * strike_vol_matrix * T_[:, np.newaxis], T_)
    total_vol = np.sqrt(total_var)

    total_vol_d = linear_interpolator(timesteps, strike_var_deriv_matrix * T_[:,np.newaxis], T_)
    total_vol_d /= total_vol

    total_vol_dd = linear_interpolator(timesteps, strike_var_dderiv_matrix * T_[:, np.newaxis], T_)
    total_vol_dd = (total_vol_dd - total_vol_d * total_vol_d) / total_vol

    num = (total_var[1:, :] - total_var[:-1, :]) / np.diff(timesteps)[:, np.newaxis]
    f1 = local_vol_x_grid_[np.newaxis, :] / total_vol - .5 * total_vol

    f2 = f1 + total_vol
    denom = (1. - total_vol_d * f1) * (1. - total_vol_d * f2) + total_vol_dd * total_vol
    grid_local_vol = np.empty_like(denom)

    grid_local_vol[0, :] = total_vol[0, :] / np.sqrt(timesteps[0])
    grid_local_vol[1:, :] = np.sqrt(num / denom[:-1, :])

    return grid_local_vol, local_vol_x_grid_