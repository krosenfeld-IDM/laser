from pathlib import Path
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt

# CPU based implementation
from idmlaser.models.numpynumba import NumbaSpatialSEIR  # noqa: I001, E402, RUF100
from idmlaser.utils import PropertySet

import sys
sys.path.append(str(Path(__file__).parents[1] / "tests"))
from scenario_engwal import initialize_engwal  # noqa: E402, I001

def get_params():

    # set the model parameters
    meta_params = PropertySet()
    meta_params.ticks = 365 * 3
    meta_params.nodes = 3 # 0 is all nodes
    meta_params.seed = 20240612
    meta_params.output = Path(__file__).parent / "outputs" # this seems dangerous...

    model_params = PropertySet()
    model_params.exp_mean = np.float32(7.0)
    model_params.exp_std = np.float32(1.0)
    model_params.inf_mean = np.float32(7.0)
    model_params.inf_std = np.float32(1.0)
    model_params.r_naught = np.float32(14.0)
    model_params.seasonality_factor = np.float32(0.1)
    model_params.seasonality_offset = np.int32(182.5)

    model_params.beta = model_params.r_naught / model_params.inf_mean


    # England and Wales Scenario
    meta_params.scenario = "engwal"

    # England and Wales network parameters (we derive connectivity from these and distance)
    net_params = PropertySet()
    net_params.a = np.float32(1.0)   # pop1 power
    net_params.b = np.float32(1.0)   # pop2 power
    net_params.c = np.float32(2.0)   # distance power
    net_params.k = np.float32(500.0) # scaling factor
    net_params.max_frac = np.float32(0.05) # max fraction of population that can migrate

    params = PropertySet(meta_params, model_params, net_params)

    return params

def run(params):

    max_capacity, demographics, initial, network = initialize_engwal(None, params, params.nodes)    # doesn't need a model, yet

    params.prng_seed = datetime.now(tz=None).microsecond  # noqa: DTZ005
    model = NumbaSpatialSEIR(params)

    # GPU based implementation with Taichi
    # from idmlaser.models import TaichiSpatialSEIR  # noqa: I001, E402, RUF100
    # model = TaichiSpatialSEIR(params)

    model.initialize(max_capacity, demographics, initial, network)

    model.run(params.ticks)

    # paramfile, npyfile = model.finalize()

    return model

def make_fig(params):
    # channels = np.load(params.output / "20240705-225923-engwal-954-7300-spatial_seir.npy")
    channels = np.load(params.output / "20240708-205224-engwal-3-1095-spatial_seir.npy")
    max_node = np.argmax(channels[0,0,:])
    plt.figure()
    plt.plot(channels[:,2,max_node], color='r', label='Infectious')
    plt.ylim(0, None)
    plt.ylabel('Infectious')
    # plt.plot(channels.sum(axis=-1)[:,2], color='r', label='Infectious')
    # add a second axis
    ax2 = plt.gca().twinx()
    ax2.plot(channels[:,0,max_node], color='b', label='Susceptible')
    ax2.set_ylim(0, None)
    ax2.set_ylabel('Susceptible')
    plt.tight_layout()
    plt.savefig(params.output / "channel.png")

    plt.figure()
    plt.plot(np.sum(channels, axis=(1,2)))
    print(f"final poulation: {np.sum(channels[-1])}")
    plt.ylabel('Population')
    plt.savefig(params.output / "population.png")


if __name__ == "__main__":

    params = get_params()

    model = run(params)
    print("POPULATION COUNT:", model.population.count)

    make_fig(params)

    print("done")