"""
Run the quickstart example
"""
import sir_numpy_c as model
import demographics_settings
import settings
import report
from measles import run_simulation

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np



def plot_sir_curves(csv_file="simulation_output.csv", node_id=0):
    # Load data from CSV file
    df = pd.read_csv(csv_file)

    # Filter data for the specified node
    node_data = df[df['Node'] == node_id]

    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plot SIR curves
    ax1.plot(node_data['Timestep'], node_data['Susceptible'], label='Susceptible', color='blue')
    ax1.plot(node_data['Timestep'], node_data['Infected'], label='Infected', color='red')
    ax1.plot(node_data['Timestep'], node_data['New Infections'], label='Incidence', color='orange')

    ax2 = ax1.twinx()
    ax2.plot(node_data['Timestep'], node_data['Recovered'], label='Recovered', color='green')
    ax2.set_ylabel('Recovered count')

    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')

    # Set plot labels and title
    plt.xlabel('Timestamp')
    plt.ylabel('Population')
    plt.title(f'SIR Curves for Node {node_id}')
    plt.legend()

    # Show the plot
    plt.savefig('SIR_curves.png')

def plot_si_orbitals(
    csv_file = "simulation_output.csv", 
    burnin = 365*4,
    node_id = 0,
    normalize = False
):

    df = pd.read_csv(csv_file)
    df_filtered = df[df['Timestep'] > int(burnin)]
    df_filtered = df_filtered[df_filtered['Node'] == node_id]
    if normalize:
        total_population = df_filtered["Susceptible"] + df_filtered["Infected"] + df_filtered["Recovered"]
        df_filtered.loc[:, "Susceptible"] = df_filtered["Susceptible"] / total_population
        df_filtered.loc[:, "Infected"] = df_filtered["Infected"] / total_population
        label = "Fraction"
    else:
        df_filtered.loc[:, "Susceptible"] = df_filtered["Susceptible"]
        df_filtered.loc[:, "Infected"] = df_filtered["Infected"]
        label = "Population"

    # Step 3: Plot the Data
    plt.figure(figsize=(8, 6))
    colors=np.linspace(0, 1, len(df_filtered["Susceptible"]))
    plt.scatter(df_filtered["Susceptible"], df_filtered["Infected"], alpha=0.5, c=colors, cmap="hsv")
    plt.title(f"Susceptible {label} vs Infected {label} (ID={node_id})")
    plt.xlabel(f"Susceptible {label}")
    plt.ylabel(f"Infected {label}")
    plt.grid(True)
    plt.savefig('SI_orbitals.png')

if __name__ == "__main__":

    # set the current working directory to the parent of this file
    import os
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # read in the pop, initialize EULA, and adjust memory footprint
    ctx = model.initialize_database() # should really be called "load model"
    ctx = model.eula_init( ctx, demographics_settings.eula_age )
    # initialize reporter
    csv_writer = report.init()
    # run the simulation
    run_simulation( ctx=ctx, csvwriter=csv_writer, num_timesteps=settings.duration )

    plot_sir_curves()

    plot_si_orbitals()
