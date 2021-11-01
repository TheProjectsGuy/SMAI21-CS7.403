# %% Import everything
import numpy as np
from matplotlib import pyplot as plt

# %% Define functions

# CDF Inverse of Exponential Distribution
def exp_cdfinv(yvals, ld):
    """
    Returns the point-wise inverse CDF for 'yvals' (whose all values 
    must be in the range (0, 1)). CDF is for Exponential distribution.
    The 'ld' is the lambda value for the exponential distribution.
    """
    return (1/ld)*np.log( 1/(1-yvals) )

# Exponential distribution function
def pdf_exp(xvals, ld):
    """
    Evaluates the value of Exponential Probability Density Function at
    given 'xvals'. The 'ld' is the lambda value for the exponential
    distribution.
    """
    return ld * np.exp(-ld*xvals)

# %% Main code
if __name__ == "__main__":
    # Random number generator
    rng = np.random.default_rng(10)
    lbda = 1.5  # Lambda value for the distribution
    N = 10000
    # Generate 10,000 samples in U[0, 1]
    yvals = rng.uniform(0, 1, N)
    xvals = exp_cdfinv(yvals, lbda)
    # Histogram
    hvals, hedges = np.histogram(xvals, 50, density=True)
    lhedges = hedges[:-1]   # Left edges
    bar_width = 0.85*(lhedges[1] - lhedges[0])
    # Exponential distribution (to cross-verify)
    x_n = np.linspace(hedges[0], hedges[-1], 100)
    exponential_vals = pdf_exp(x_n, lbda)
    # Plot everything
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.bar(lhedges, hvals, align='edge', width=bar_width, \
        label="inv CDF")
    ax.plot(x_n, exponential_vals, 'r--', label="Exp Dist")
    ax.set_title("Exponential Distribution using inverse CDF")
    ax.legend()
    ax.set_xlabel("x")
    ax.set_ylabel("p(x)")
    fig.savefig("plot_p6_exp.png", dpi=600)
    plt.show()

# %%
