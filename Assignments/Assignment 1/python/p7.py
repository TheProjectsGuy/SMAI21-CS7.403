# %% Import everything
import numpy as np
from matplotlib import pyplot as plt

# %% Define functions

# Run the experiment once
def experiment_trial(rng: np.random.Generator, N = 500):
    """
    Runs the experiment: Generates 'N' random numbers from Uniform
    Probability Density Function (low = 0, high = 1), adds them all up
    and returns the resultant sum. The function requires the 'rng'
    which is the random number generator object (numpy.random)
    """
    # Generate N samples
    samples = rng.uniform(0, 1, N)
    return np.sum(samples)  # Return their sum

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
    N = 500 # n value for each experiment (number of samples of U)
    N_iter = 50000   # Number of times the experiment must be run
    # Rnadom Number Generator
    rng = np.random.default_rng(10)
    exp_results = []
    for _ in range(N_iter):
        exp_results.append(experiment_trial(rng, N))
    # Convert results to numpy array
    exp_results = np.array(exp_results, dtype=float)
    # Generate normalized histogram from experiment results
    hvals, hedges = np.histogram(exp_results, 50)
    hvals_n = hvals/np.sum(np.diff(hedges)*hvals)   # Normalize hist.
    ledges = hedges[0:-1]
    # Approximating a normal distribution
    x_n = np.linspace(hedges[0], hedges[-1], 100)
    app_norm_vals = pdf_normal(x_n, N/2, N/12)
    # Plot the resultant histogram
    fig = plt.figure()
    ax = fig.add_subplot()
    bin_width = 0.85 * (ledges[1]-ledges[0])
    ax.bar(ledges, hvals_n, width=bin_width, align="edge", \
        label="Irwin-Hall")
    ax.plot(x_n, app_norm_vals, 'r--', label="Normal")
    ax.set_title("Irwin-Hall distribution " + \
        r"$X \rightarrow \sum^{" + str(N) + r"}_{k=1} U(0,1) $")
    ax.set_xlabel(r"$X$")
    ax.set_ylabel(r"Normalized Histogram of $X$ sampled " \
        + str(N_iter) + r" times")
    ax.grid()
    ax.legend()
    fig.savefig("plot_p7.png", dpi=600)
    plt.show()

# %%
