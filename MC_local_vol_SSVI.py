

from scipy.stats import norm
import numpy as np

from cone import linear_slice_interpolator
from local_volatility_generator import local_volatility_grid
import matplotlib.pyplot as plt
import time

def porfolio_black_call(strike, forwards, total_volatilities):
    d1 = np.log(forwards/strike)/total_volatilities + .5 * total_volatilities
    d2 = d1-total_volatilities
    out = forwards * norm.cdf(d1) - strike * norm.cdf(d2)
    return out


def phi(theta, params):
    gamma, eta, sigma, rho = params
    return eta / pow(theta, gamma)


def SSVI(x, t, params):
    gamma, eta, sigma, rho = params
    theta = sigma * sigma * t
    p = phi(theta, params)
    return 0.5 * theta * (1. + rho * p * x + np.sqrt((p * x + rho) * (p * x + rho) + 1. - rho * rho))


expiry = 1.

xx, TT = np.linspace(-1., 1., 50), np.linspace(0.1, expiry, 10)

strikes = np.linspace(0.6,1.4,50)
xx = np.log(strikes)

timesteps = np.linspace(0,expiry,int(expiry*252/5))
timesteps = np.array(sorted(set(timesteps.tolist()+TT.tolist())))

sigma, gamma, eta, rho = 0.2, 0.44, 0.5, -0.7
#sigma, gamma, eta, rho = 0.2, 0.0, 0., 0

params = gamma, eta, sigma, rho
print("Consistency check to avoid static arbitrage: ", (gamma - 0.25*(1.+np.abs(rho))>0.))
impliedSSVIgridvol = np.array([[np.sqrt(SSVI(x,t,params)/t) for x in xx] for t in TT])

fig = plt.figure(figsize=(8,5))
ax = fig.gca(projection='3d')
xxx, TTT = np.meshgrid(xx, TT)
localVolatilityeSSVI = impliedSSVIgridvol
ax.plot_surface(xxx, TTT, localVolatilityeSSVI, cmap=plt.cm.jet, rstride=1, cstride=1, linewidth=0)
ax.set_xlabel("Log-moneyness")
ax.set_ylabel("Maturity")
ax.set_zlabel("Local volatility")
ax.set_title("SSVI implied volatility surface")
plt.show()

fwds = np.ones(len(TT))

np.random.seed(10)
strike = 0.9
#strike = 1.

dt = np.diff(timesteps)
sqrt_dt = np.sqrt(dt)
minus_half_sqrt_dt = -.5*sqrt_dt
n_paths = 100000
log_spot_diffusion = np.zeros(n_paths)


start_time = time.time()

local_vol_grid, local_vol_x_axis = local_volatility_grid(TT, strikes, impliedSSVIgridvol, fwds, timesteps[1:])
#local_vol_grid, local_vol_x_axis = local_volatility_grid(TT, None, impliedSSVIgridvol, fwds, timesteps[1:], xx)
end_calib_time = time.time()

for kk in range(len(timesteps)-1):
    local_volatility = linear_slice_interpolator(log_spot_diffusion, local_vol_grid[kk, :], local_vol_x_axis)
    log_spot_diffusion += sqrt_dt[kk] * local_volatility * np.random.normal(minus_half_sqrt_dt[kk] * local_volatility, 1., n_paths)



spot = np.exp(log_spot_diffusion) #This is the 'M' process
mc_price = np.average(np.maximum(spot-strike, 0.))
end_time = time.time()
ssvi_total_vol = np.sqrt(SSVI(np.log(strike), timesteps[-1], params))
cf_price = porfolio_black_call(strike,1., ssvi_total_vol)

print('Error in % ', 100.*(cf_price/mc_price-1.), 'calibration took ', end_calib_time - start_time, ' MC sim took ', end_time-end_calib_time)
