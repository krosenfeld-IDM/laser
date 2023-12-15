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

SEED = np.uint32(20231205)
POP_SIZE = np.uint32(1_000_000)
INIT_INF = np.uint32(10000)

_prng = np.random.default_rng(seed=SEED)

# R_NAUGHT = np.float32(2.5)
MEAN_INF = np.float32(5)
STD_INF = np.float32(1)
# BETA = np.float32(R_NAUGHT / MEAN_INF)

TIMESTEPS = np.uint32(128)

def test_sir(r_naught):
    """
    Run SIR agent-based model and return attack fraction
    """

    DOB_TYPE_NP = np.int32
    SUSCEPTIBILITY_TYPE_NP = np.float32
    SUSCEPTIBILITY_TYPE_NB = nb.float32
    ITIMER_TYPE_NP = np.uint8
    ITIMER_TYPE_NB = nb.uint8
    BETA = np.float32(np.float32(r_naught) / MEAN_INF)

    # print(f"Creating a well-mixed SIR community with {POP_SIZE:_} individuals.")
    community = abc(POP_SIZE, **{"beta": BETA})
    community.add_property("dob", dtype=DOB_TYPE_NP, default=0)
    community.add_property("susceptibility", dtype=SUSCEPTIBILITY_TYPE_NP, default=1.0)
    community.add_property("itimer", dtype=ITIMER_TYPE_NP, default=0)

    # initialize the dob property to a random value between 0 and 100*365
    community.dob = -_prng.integers(0, 100*365, size=community.count, dtype=DOB_TYPE_NP)

    # # initialize the susceptibility property to a random value between 0.0 and 1.0
    # community.susceptibility = _prng.random_sample(size=community.count)

    # select INIT_INF individuals at random and set their itimer to normal distribution with mean 5 and std 1
    community.itimer[_prng.choice(community.count, size=INIT_INF, replace=False)] = _prng.normal(MEAN_INF, STD_INF, size=INIT_INF).round().astype(ITIMER_TYPE_NP)

    community.susceptibility[community.itimer > 0] = 0.0

    @nb.njit((ITIMER_TYPE_NB[:], nb.uint32), parallel=True)
    def infection_update_inner(timers, count):
        for i in nb.prange(count):
            if timers[i] > 0:
                timers[i] -= 1
        return

    def infection_update(community, _timestep):

        # community.itimer[community.itimer > 0] -= 1
        infection_update_inner(community.itimer, community.count)

        return

    community.add_step(infection_update)

    @nb.njit((     SUSCEPTIBILITY_TYPE_NB[:], ITIMER_TYPE_NB[:], nb.uint32, nb.float32), parallel=True)
    def transmission_inner(susceptibility,               itimer,     count,       beta):
        contagion = (itimer != 0).sum()
        force = beta * contagion * (1.0 / count)
        for i in nb.prange(count):
            if np.random.random_sample() < (force * susceptibility[i]):
                susceptibility[i] = 0.0
                itimer[i] = ITIMER_TYPE_NP(np.round(np.random.normal(MEAN_INF, STD_INF)))

        return

    def transmission(community, _timestep):

        transmission_inner(community.susceptibility, community.itimer, community.count, community.beta)

        return

    community.add_step(transmission)

    @nb.njit((SUSCEPTIBILITY_TYPE_NB[:], nb.uint32), parallel=True)
    def vaccinate_inner(susceptibility, count):
        for i in nb.prange(count):
            if np.random.binomial(1, 0.6) == 1:
                susceptibility[i] = 0.0
        return

    def vaccinate(community, timestep):

        if timestep == 30:
            # do a binomial draw with probability 0.6 and set the susceptibility to 0.0 for those individuals
            # community.susceptibility[np.random.binomial(1, 0.6, size=community.count, dtype=np.bool)] = 0.0
            vaccinate_inner(community.susceptibility, community.count)

        return

    # community.add_step(vaccinate)

    results = np.zeros((TIMESTEPS+1, 4), dtype=np.uint32)

    def record(timestep, community, results):

        """Record the state of the community at the current timestep"""

        results[timestep,0] = timestep
        results[timestep,1] = (community.susceptibility > 0.0).sum()
        results[timestep,2] = (community.itimer > 0).sum()
        results[timestep,3] = ((community.susceptibility == 0.0) & (community.itimer == 0)).sum()

        return

    record(0, community=community, results=results)

    start = datetime.now()
    # for timestep in tqdm(range(TIMESTEPS)):
    for timestep in range(TIMESTEPS):
        community.step(timestep)
        record(timestep+1, community=community, results=results)

    # finish = datetime.now()
    # print(f"elapsed time: {finish - start}")

    # attack fraction is total (pop - susceptibles) / pop
    print(results[-1,1])
    return np.float32(1.0)-np.float32(results[-1,1])/np.float32(POP_SIZE)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--generate_samples", type=int, default=0)

    args = parser.parse_args()

    if args.generate_samples:
        Zs = []
        R_NAUGHTs = []
        for r_naught in tqdm(np.random.uniform(size=args.num_samples, low=0.5, high=1.75)):
            z = test_sir(r_naught)
            Zs.append(z)
            R_NAUGHTs.append(r_naught)

        # save results for plotting
        df = pl.DataFrame({"R_NAUGHT": R_NAUGHTs, "Z": Zs})
        df.write_csv("KMlimit.csv")
    else:
        df = pl.read_csv("KMlimit.csv")
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
    ax.scatter(df["R_NAUGHT"], df["Z"], marker='o', label='Agent-based')

    ax.legend(loc='upper left')
    ax.set_xlim( 0.5, 1.75 )
    ax.set_ylim(-0.01, 0.81)
    ax.set_yticks(ax.get_yticks())
    ax.set_yticklabels(["{:0.0f}%".format(100*s) for s in ax.get_yticks()])
    ax.set_xlabel('Reproductive Number')
    ax.set_ylabel('Population Infected')
    # ax.set_ylim(-0.01, 0.81)
    fig.tight_layout()
    plt.savefig('KMlimit.png', transparent=0)
