"""
THe Kormac-Mckendrick limit predicts the size of an outbreak based on the R0 values for an
SIR model.
"""

from argparse import ArgumentParser
from datetime import datetime

import numba as nb
import numpy as np
import polars as pl
from tqdm import tqdm
from scipy.integrate import odeint

from idmlaser.community.homogeneous_abc import HomogeneousABC as abc

import sys
sys.path.append("./")
from test_agentseir import test_seir

class ODict():
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--num_samples", type=int, default=50)
    parser.add_argument("--generate_samples", type=int, default=0)

    args = parser.parse_args()

    pdict = dict(
        timesteps=np.uint32(128),
        pop_size=np.uint32(100_000),
        exp_mean=np.float32(4),
        exp_std=np.float32(1),
        inf_mean=np.float32(5),
        inf_std=np.float32(1),
        initial_inf=np.uint32(10),
        r_naught=np.float32(2.5),
        vaccinate=False,
        poisson=True,
        seed=np.uint32(20231205),
        masking=False,
        filename="seir.csv"
    )
    params = ODict(**pdict)
    params.beta = np.float32(params.r_naught / params.inf_mean)

    if args.generate_samples:
        Zs = []
        R_NAUGHTs = []
        for r_naught in tqdm(np.linspace(0.5, 1.75, args.num_samples)):
            params.r_naught = r_naught
            params.beta = np.float32(params.r_naught / params.inf_mean)
            test_seir(params)
            df = pl.read_csv(params.filename)
            z = 1 - (df["susceptible"][-1] / params.pop_size)
            Zs.append(z)
            R_NAUGHTs.append(r_naught)

        # save results for plotting
        df = pl.DataFrame({"R_NAUGHT": R_NAUGHTs, "Z": Zs})
        df.write_csv("KMlimit_seir.csv")
    else:
        df = pl.read_csv("KMlimit_seir.csv")
        print(df.head())
    # plot results againt KMlimit
    import scipy.optimize as spopt
    import matplotlib.pyplot as plt

    # Reference trajectory (Kermack-McKendric analytic solution)
    def KMlimt (x,R0):
        return 1-x-np.exp(-x*R0)    

    fig = plt.figure()
    ax = plt.gca()
    xref = np.linspace(1.01,2.0,200)
    yref = np.zeros(xref.shape)
    for k1 in range(yref.shape[0]):
        yref[k1] = spopt.brentq(KMlimt, 1e-5, 1, args=(xref[k1]))    

    ax.plot(np.concatenate((np.linspace(0.5,1.0,5), xref)),
            np.concatenate((np.zeros(5),yref)),
        '-',color='k', lw=5.0,label='Analytic Solution')
    # for i in [10, 10000]:
    #     df = pl.read_csv("KMlimit_{}.csv".format(i))
    #     ax.scatter(df["R_NAUGHT"], df["Z"], marker='o', label='Init: {}'.format(i))
    ax.scatter(df["R_NAUGHT"], df["Z"], marker='o', label='Agent-based', zorder=4)

    ax.legend(loc='upper left')
    ax.set_xlim( 0.5, 1.75 )
    ax.set_ylim(-0.01, 0.81)
    ax.set_yticks(ax.get_yticks())
    ax.set_yticklabels(["{:0.0f}%".format(100*s) for s in ax.get_yticks()])
    ax.set_xlabel('Reproductive Number')
    ax.set_ylabel('Population Infected')
    # ax.set_ylim(-0.01, 0.81)
    fig.tight_layout()
    plt.savefig('KMlimit_seir.png', transparent=0)
