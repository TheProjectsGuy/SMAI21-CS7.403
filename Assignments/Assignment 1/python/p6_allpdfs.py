# pylint: skip-file
# %% Import everything
import numpy as np
from matplotlib import pyplot as plt
# PDFs in earlier codes
from p6_normal import pdf_normal
from p6_rayleigh import pdf_rayleigh
from p6_exp import pdf_exp

# %% Main code
if __name__ == "__main__":
    xlim = [-5, 6]
    n_m, n_sigsq = 0, 1.25 # Mean and Variance of Normal Distribution
    r_sigsq = 1.25 # Sigma squared of Rayleigh Distribution
    e_lambda = 1.5  # Lambda value of Exponential Distribution
    # X values
    xvals = np.linspace(xlim[0], xlim[1], 100)
    # All distribution values
    normal_vals = pdf_normal(xvals, n_m, n_sigsq)
    xvpos = xvals[xvals >= 0]   # Only positive side of X
    rayleigh_vals = pdf_rayleigh(xvpos, r_sigsq)
    exp_vals = pdf_exp(xvpos, e_lambda)
    # Plot names
    normal_name = fr"$N(x;\mu={n_m}, \sigma^2={n_sigsq})$"
    rayleigh_name = fr"$R(x; \sigma^2={r_sigsq})$"
    exponential_name = fr"$E(x; \lambda = {e_lambda})$"
    # Plot everything
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.axvline(0, c='k', ls='-.')
    ax.plot(xvals, normal_vals, '--', label=normal_name)
    ax.plot(xvpos, rayleigh_vals, '--', label=rayleigh_name)
    ax.plot(xvpos, exp_vals, '--', label=exponential_name)
    ax.set_xlim(xlim)
    ax.set_ylim([0, 0.8])
    ax.set_title("Different Probability Density Functions")
    ax.legend()
    ax.grid()
    fig.savefig("plot_p6_all.png", dpi=600)
    plt.show()

# %%
