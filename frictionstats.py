"""
Compute confidence interval for a quantile.

Suppose I'm interested in estimating the 37th percentile.  The
empirical CDF gives me one estimate for that.  I'd like
to get a confidence interval: I'm 90% confident that the 37th percentile
lies between X and Y.

You can compute that with two calls to the following function
(supposing you're interested in [5%-95%] range) by something like the
following:
n = len(sorted_data)
X_index = CDF_error(n,0.37,0.05)
Y_index = CDF_error(n,0.37,0.95)
X=sorted_data[X_index]
Y=sorted_data[Y_index]
90% confidence interval is [X,Y]
"""
from scipy.stats import binom, beta
from scipy import interpolate


# The beta distribution is the correct (pointwise) distribution
# across *quantiles* for a given *data point*; if you're not
# sure, this is probably the estimator you want to use.
def CDF_error_beta(n, target_quantile, quantile_quantile):
    k = target_quantile * n
    return (beta.ppf(quantile_quantile, k, n + 1 - k))


# Boot strapping can give you a distribution across *values* for a given
# *quantile*.  Warning: Although it is asymptotically correct for quantiles
# in (0,1), bootstrapping fails for the extreme values (i.e. for quantile=0
# or 1.  Moreover, you should be suspicious of confidence intervals for
# points within 1/sqrt(data size).
def CDF_error_analytic_bootstrap(n, target_quantile, quantile_quantile):
    target_count = int(target_quantile * float(n))

    # Start off with a binary search
    small_ind = 0
    big_ind = n - 1
    small_prob = 1 - binom.cdf(target_count, n, 0)
    big_prob = 1 - binom.cdf(target_count, n, float(big_ind) / float(n))

    while big_ind - small_ind > 4:
        mid_ind = (big_ind + small_ind) / 2
        mid_prob = 1 - binom.cdf(target_count, n, float(mid_ind) / float(n))
        if mid_prob > quantile_quantile:
            big_prob = mid_prob
            big_ind = mid_ind
        else:
            small_prob = mid_prob
            small_ind = mid_ind

            # Finish it off with a linear search
    prob_closest = -100
    for p_num in xrange(small_ind, big_ind + 1):
        p = float(p_num) / float(n)
        coCDF_prob = 1 - binom.cdf(target_count, n, p)
        if abs(coCDF_prob - quantile_quantile) < abs(prob_closest - quantile_quantile):
            prob_closest = coCDF_prob
            prob_index = p_num

    return (prob_index)


# Compute Dvoretzky-Kiefer-Wolfowitz confidence bands.
def CDF_error_DKW_band(n, target_quantile, quantile_quantile):
    # alpha is the total confidence interval size, e.g. 90%.
    alpha = 1.0 - 2.0 * np.abs(0.5 - quantile_quantile)
    epsilon = np.sqrt(np.log(2.0 / alpha) / (2.0 * float(n)))
    if quantile_quantile < 0.5:
        return (max((0, target_quantile - epsilon)))
    else:
        return (min((1, target_quantile + epsilon)))






