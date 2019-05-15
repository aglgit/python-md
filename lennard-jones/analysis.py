import matplotlib.pyplot as plt
import seaborn as sns
from amp.analysis import read_trainlog

sns.set()

log = read_trainlog("amp-log.txt")
d = log["convergence"]

steps = d["steps"]
e_rmse = d["emrs"]
f_rmse = d["fmrs"]
loss = d["costfxns"]

plt.semilogy(steps, e_rmse, label="Energy RMSE")
plt.semilogy(steps, f_rmse, label="Force RMSE")
plt.semilogy(steps, loss, label="Loss function")

plt.title("Energy and force Root Mean Square Error")
plt.legend()
plt.xlabel("Steps")
plt.ylabel("Error [eV, eV/Å]")
plt.savefig("loss.png")
