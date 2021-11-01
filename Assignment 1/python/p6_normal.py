# pylint: skip-file
# %% Import everything
import numpy as np
from matplotlib import pyplot as plt
from scipy import special as sps

# %% Define functions

# CDF Inverse of Normal Distribution
def normal_cdfinv(yvals, mu, sig_sq):
    """
    Returns the point-wise inverse CDF for 'yvals', all values in the
    range (0, 1). The mean is 'mu' and variance is 'sig_sq'
    """
    x = mu + np.sqrt(2 * sig_sq) * sps.erfinv(2 * yvals - 1)
    return x

# Normal distribution function
def pdf_normal(xvals, mu, sig_sq):
    """
    Evaluates the value of the Normal Probability Density Function at
    the given 'xvals'. The value 'mu' is mean and variance is 'sig_sq'
    """
    cv = 1/np.sqrt(2*np.pi*sig_sq)
    return cv * np.exp(-(xvals-mu)**2 / (2*sig_sq))

# %% Main code
if __name__ == "__main__":
    # Random number generator
    rng = np.random.default_rng(10)
    mu, sigma = 0, 3    # Mean and standard deviation
    N = 10000
    # Generate 10000 samples in U[0, 1]
    yvals = rng.uniform(0, 1, N)
    xvals = normal_cdfinv(yvals, mu, sigma**2)
    # Histogram
    hvals, hedges = np.histogram(xvals, 50, density=True)
    lhedges = hedges[:-1]   # Left edges
    bar_width = 0.85*(lhedges[1] - lhedges[0])
    # Normal distribution (to cross-verify)
    x_n = np.linspace(hedges[0], hedges[-1], 100)
    norm_vals = pdf_normal(x_n, mu, sigma**2)
    # Plot everything
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.bar(lhedges, hvals, align='edge', width=bar_width, \
        label="inv CDF")
    ax.plot(x_n, norm_vals, 'r--', label="Normal Dist")
    ax.set_title("Normal Distribution using inverse CDF")
    ax.legend()
    ax.set_xlabel("x")
    ax.set_ylabel("p(x)")
    fig.savefig("plot_p6_normal.png", dpi=600)
    plt.show()

# %%
