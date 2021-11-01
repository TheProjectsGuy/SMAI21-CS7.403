# %% Import everything
import numpy as np
from matplotlib import pyplot as plt

# %% Define functions

# CDF Inverse of Rayleigh Distribution
def rayleigh_cdfinv(yvals, sig_sq):
    """
    Returns the point-wise inverse CDF for 'yvals' (whose all values 
    must be in the range (0, 1)). CDF is for Rayleigh distribution.
    The 'sig_sq' is squared of sigma (a parameter for the 
    distribution).
    """
    return np.sqrt( 2*sig_sq*np.log(1/(1-yvals)) )

# Rayleigh distribution function
def pdf_rayleigh(xvals, sig_sq):
    """
    Evaluates the value of Rayleigh Probability Density Function at
    given 'xvals'. The value for 'sig_sq' is the squared of sigma (a
    parameter for the distribution).
    """
    return (xvals/sig_sq) * np.exp( -(xvals**2)/(2*sig_sq) )

# %% Main code
if __name__ == "__main__":
    # Random number generator
    rng = np.random.default_rng(10)
    sigma = 1.0
    N = 10000
    # Generate 10,000 samples in U[0, 1]
    yvals = rng.uniform(0, 1, N)
    xvals = rayleigh_cdfinv(yvals, sigma**2)
    # Histogram
    hvals, hedges = np.histogram(xvals, 50, density=True)
    lhedges = hedges[:-1]   # Left edges
    bar_width = 0.85*(lhedges[1] - lhedges[0])
    # Rayleigh distribution (to cross-verify)
    x_n = np.linspace(hedges[0], hedges[-1], 100)
    rayleigh_vals = pdf_rayleigh(x_n, sigma**2)
    # Plot everything
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.bar(lhedges, hvals, align='edge', width=bar_width, \
        label="inv CDF")
    ax.plot(x_n, rayleigh_vals, 'r--', label="Rayleigh Dist")
    ax.set_title("Rayleigh Distribution using inverse CDF")
    ax.legend()
    ax.set_xlabel("x")
    ax.set_ylabel("p(x)")
    fig.savefig("plot_p6_rayleigh.png", dpi=600)
    plt.show()

# %%
