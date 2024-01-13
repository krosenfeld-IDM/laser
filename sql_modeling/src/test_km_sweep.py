"""
Sweep across base infectivity values to look at final infected fraction
"""

import pdb
import numpy as np
import polars as pl
import matplotlib.pyplot as plt
from tqdm import tqdm
from icecream import ic
import settings
import report
import sir_numpy as model

def run_simulation(ctx, csvwriter, num_timesteps):
    """
    Run one interation of the simulation

    Parameters
    ----------
    ctx : dict
        The database context
    csvwriter : csv.writer
        The csv writer object
    num_timesteps : int
        The number of timesteps to run
    """
    currently_infectious, currently_sus, cur_reco = model.collect_report( ctx )
    report.write_timestep_report( csvwriter, 0, currently_infectious, currently_sus, cur_reco )

    for timestep in tqdm(range(1, num_timesteps + 1)):

        # We almost certainly won't waste time updating everyone's ages every timestep but this is 
        # here as a placeholder for "what if we have to do simple math on all the rows?"
        #ctx = model.update_ages( ctx )

        # We should always be in a low prev setting so this should only really ever operate
        # on ~1% of the active population
        ctx = model.progress_infections( ctx )

        # The perma-immune should not consume cycles but there could be lots of waning immune
        ctx = model.progress_immunities( ctx )

        # The core transmission part begins
        new_infections = model.calculate_new_infections( ctx, currently_infectious, currently_sus )

        # TBD: for loop should probably be implementation-specific
        ctx = model.handle_transmission( ctx, new_infections )

        # Sets the new infection timers globally
        ctx = model.add_new_infections( ctx )

        # Transmission is done, now migrate some. Only infected?
        ctx = model.migrate( ctx, timestep, num_infected=sum(currently_infectious.values()) )
        #conn.commit() # deb-specific

        # Report
        currently_infectious, currently_sus, cur_reco = model.collect_report( ctx )
        report.write_timestep_report( csvwriter, timestep, currently_infectious, currently_sus, cur_reco )

    
def run_km_sweep(num_samples:int=5):
    """
    Sweep over R0 values and calculate the total fraction infected
    """

    # mean infection period is 9 days (https://github.com/krosenfeld-IDM/laser/blob/e0310fcd47d1345f2d834b2dbfb589c17b0f13db/sql_modeling/src/sir_numpy.py#L302)
    mean_infection_period = 9
    # R0s = np.linspace(0.5, 1.6, num=num_samples, endpoint=True)
    R0s = [0]
    Zs = []
    for R0 in R0s:

        # update the infectivity
        settings.base_infectivity = R0 / settings.pop / settings.num_nodes / mean_infection_period

        # Initialize the 'database' (or load the dataframe/csv)
        # ctx might be db cursor or dataframe or dict of numpy vectors
        ctx = model.initialize_database()

        # initialize the reporter
        csvwriter = report.init()

        # run the simulation
        run_simulation(ctx, csvwriter, settings.duration)

        # write results to disk
        report.close()

        # grab the results
        df = pl.read_csv(settings.report_filename)        

        # calculate the final fraction infected
        Zs.append(1 - df[-1,2] / settings.pop)


    df = pl.DataFrame({'R0': R0s, 'Z': Zs})

    df.write_csv('km_sweep_results.csv')

    return 


def plot_km_sweep():
    """
    Plot results of run_km_sweep
    """

    df = pl.read_csv('km_sweep_results.csv')

    plt.plot(df['R0'], df['Z'], 'o')
    plt.xlabel('R0')
    plt.ylabel('Fraction infected')
    plt.title('Fraction infected vs R0')
    plt.ylim(0, 1)
    plt.savefig('km_sweep_results.png')

    return

if __name__ == "__main__":
    run_km_sweep()
    plot_km_sweep()
    print('done')