import numpy as np
import matplotlib.pyplot as plt
import os

# get directory
path = os.path.dirname(os.path.abspath(__file__))
path = os.path.join(path, "..", '..', '..', "run_data", "potentials", "type_a2")
file_names = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]

for i in range(0, 13):
    file_path = os.path.join(path, 'potential_' + str(i) + '.csv')
    data = np.genfromtxt(file_path, delimiter=',', skip_header=True).T

    plt.figure(dpi=100)
    plt.plot(data[0], data[1] / (246.0**4), label=r"$v_{\mathrm{eff}}$")
    plt.plot(data[0], data[2] / (246.0**4), label=r"$v_{\mathrm{tree}}$")
    plt.xlabel(r"$t$", fontsize=16)
    plt.ylabel(r"$V_{\mathrm{eff}} / \mu^4$", fontsize=16)
    plt.legend()
    plt.tight_layout()
    plt.savefig('potential_' + str(i) + ".pdf")
    plt.close('all')
