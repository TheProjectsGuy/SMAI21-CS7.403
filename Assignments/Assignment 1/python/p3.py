# %% Import everything
import numpy as np
from matplotlib import pyplot as plt

# %% Declare functions

# Generate uniform distribution function
def gen_uni_dist(a, b):
    """
    Generates a callable uniform distribution for a continuous random
    variable. Pass the 'a' and 'b' values ('b' > 'a').
    """
    if (a > b):
        return gen_uni_dist(b, a)
    cv = (1/(b-a))
    ud_func = lambda x: cv if a <= x and x <= b else 0
    return ud_func

# Generate normal distribution function
def gen_norm_dist(mu, si2):
    """
    Generates a callable normal distribution for a continuous random 
    variable. Pass the 'mu' (mean) and 'si2' (variance = square of the
    standard deviation)
    """
    nd_func = lambda x: 1/( np.sqrt(2*np.pi*si2) ) * \
        np.exp( -(x-mu)**2/(2*si2) )
    return nd_func

# %% Plot everything
if __name__ == "__main__":
    # Configurations
    a, b = 2, 4
    mu, si2 = (a+b)/2, ((b-a)**2)/12
    xlim = [0, 5]
    # Actual code to generate data to be plotted
    xvals = np.linspace(xlim[0], xlim[1], 100)
    uni_dist = gen_uni_dist(a, b)
    uni_vals = np.array([uni_dist(xv) for xv in xvals])
    norm_vals = gen_norm_dist(mu, si2)(xvals)
    # Plot everything
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.plot(xvals, uni_vals, 'b--', label='Uniform')
    ax.plot(xvals, norm_vals, 'r--', label='Normal')
    ax.legend()
    ax.set_title(fr"$\mu={mu:.2f}\,\,and\,\,\sigma^2={si2:.3f}$")
    fig.savefig("plot_p3.png", dpi=600)
    plt.show()

# %%
