import numpy as np
import matplotlib.pyplot as plt
import os

# get directory
path = os.path.dirname(os.path.abspath(__file__))
path = os.path.join(path, "..", '..', "run_data", "potentials")
file_names = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]

for file_name in file_names:
    file_path = os.path.join(path, file_name)
    data = np.genfromtxt(file_path, delimiter=',', skip_header=True).T

    plt.figure(dpi=100)
    plt.plot(data[0], data[1] / (246.0**4), label=r"$v_{\mathrm{eff}}$")
    plt.plot(data[0], data[2] / (246.0**4), label=r"$v_{\mathrm{tree}}$")
    plt.xlabel(r"$t$", fontsize=16)
    plt.ylabel(r"$V_{\mathrm{eff}} \ (\mathrm{GeV}^4)$", fontsize=16)
    plt.legend()
    plt.tight_layout()
    plt.savefig(file_name.split(".")[0] + ".pdf")
    plt.close('all')
