import sys

sys.path.insert(0, "../tools")

import matplotlib.pyplot as plt
from analysis import Analyzer

anl = Analyzer(save_interval=100)
theta, adf = anl.calculate_adf("trajs/training.traj", r_cut=6.0, nbins=100)
plt.plot(theta[:-1], adf)
plt.show()
