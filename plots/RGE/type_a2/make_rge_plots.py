import numpy as np
import matplotlib.pyplot as plt
import os

# get directory
path = os.path.dirname(os.path.abspath(__file__))
path = os.path.join(path, "..", '..', '..', "run_data", "RGE", 'type_a2')
# file_names = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]

for i in range(0, 10):
    file_path = os.path.join(path, 'rge_' + str(i) + '.csv')
    data = np.genfromtxt(file_path, delimiter=',', skip_header=True).T

    plt.figure(dpi=100)
    plt.plot(data[0], data[1] / (246.0**4), label="normal")
    plt.plot(data[0], data[2] / (246.0**4), label="charge-breaking")
    plt.xlabel(r"$\mu \ (\mathrm{GeV})$", fontsize=16)
    plt.ylabel(r"$V_{\mathrm{eff}} / 246^4$", fontsize=16)
    plt.legend()
    plt.tight_layout()
    plt.savefig('rge_' + str(i) + ".pdf")
    plt.close('all')
