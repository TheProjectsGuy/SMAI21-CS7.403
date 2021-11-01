# pylint: skip-file
# %% Import everything
import numpy as np
from matplotlib import pyplot as plt
from scipy import special as sps

# %%
xvals = np.linspace(-3, 3)
yvals = sps.erf(xvals)
plt.plot(xvals, yvals, 'b.')

# %%
xvals = np.linspace(-0.99, 0.99)
yvals = sps.erfinv(xvals)
plt.plot(xvals, yvals, 'b.')

# %%
