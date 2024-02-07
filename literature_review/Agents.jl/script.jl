"""
"Create an agent based SEIR disease model with 1 million agents distributed unevenly across 25 nodes with 
an initial outbreak of 100 agents. β and mean incubation period are inputs to the model. Run for 1 simulated 
year and plot the aggregated SEIR curves for the total population and the SEIR curves for each node in a 5 by 5 grid of plots."
"""

using Agents
using Agents.Graphs
using StatsBase


#
γ = 1/10 # recovery rate; 1 / average duration of infection
β = # R_0 / gamma

# Agent structure
@agent PoorSoul GraphAgent begin
    # id, pos, days_infected, s tatus
    days_infected::Int  # number of days since is infected
    status::Symbol  # 1: S, 2: I, 3:R
end

"""
Initialize the model by:
1. Setting up the spatial structure. 

"""
function initialize(num_agents=100)

    # Spatial setup (NxN grid)
    grid_dims = (5, 5)
    space = GraphSpace(grid(grid_dims))

    # Fixed properties of the model
    properties = dict(
        β <: β
    )

    # Initialize the model
    model = ABM(PoorSoul, space; properties, rng)

    # Add initial individuals
    W = rand(nv(space.graph))
    W ./= sum(W)
    for p in 1:num_agents
        node = sample(1:N, Weights(W))
        add_agent!(node, model, 0, :S) # Susceptible
    end

    # # add initial outbreak
    # for city in 1:C
    #     inds = ids_in_position(city, model)
    #     for n in 1:Is[city]
    #         agent = model[inds[n]]
    #         agent.status = :I # Infected
    #         agent.days_infected = 1
    #     end
    # end    

    return model
end

model = initialize()