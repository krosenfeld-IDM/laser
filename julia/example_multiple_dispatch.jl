"""
Example exploring multiple dispatch.
- This is probably not compact enough in memory.
"""

module ABM

export AbstractAgent, SIRAgent, SEIRAgent, force_of_infection, infect!, transmit!

parameters = Dict(
    :num_agents => 1000,
    :num_spatial_units => 1,
    :r0 => 2.5,
    :infectious_period => 5,
    :num_timesteps => 100
)

# define an abstract type Agent
abstract type AbstractAgent end

# SIR Agent
mutable struct SIRAgent <: AbstractAgent
    state::Symbol
    timer1::Int
end
SIRAgent() = SIRAgent(:S, 0)

# SIR Agent
mutable struct SEIRAgent <: AbstractAgent
    state::Symbol    
    timer1::Int
    timer2::Int
end
SEIRAgent() = SEIRAgent(:S, 0, 0)

"""
Infect the agent by setting the clock on their initial state
"""
function infect!(agent::AbstractAgent, initial_state::Symbol, timer::Int)
    agent.state = initial_state
    agent.timer1 = timer + 1 # +1 for initial state
end
# hard code the length of the initial state for this example
infect!(agent::SIRAgent) = infect!(agent, :I, parameters[:infectious_period])
infect!(agent::SEIRAgent) = infect!(agent, :E, 3)

function recover!(agent::AbstractAgent)
    agent.state = :R
end

"""
The force of infection is the same whether you are using a SIR or SEIR model
"""
function force_of_infection(agents::Matrix{T}) where T <: AbstractAgent
    return count(x -> x.state == :I, agents) * (parameters[:r0] / length(agents) / parameters[:infectious_period])
end

"""
Define a Julia function transmit which is a function of a matrix of SIR agents
"""
function transmit!(agents::Matrix{T}) where T <: AbstractAgent

    # the force of infection is the number of infectious agents divided by the total number of agents
    foi = force_of_infection(agents)

    # loop over agents and set their state to I with probability foi
    for i in eachindex(agents)
        if agents[i].state == :S && rand() < foi
            infect!(agents[i])
        end
    end
end

"""
"""
function step!(agents::Matrix{SIRAgent})
    for i in eachindex(agents)
        if agents[i].timer1 > 0
            agents[i].timer1 -= 1
            # check recovery
            if (agents[i].timer1 == 0)
                recover!(agents[i])
            end                    
        end
    end
end

function step!(agents::Matrix{SEIRAgent})
    for i in eachindex(agents)
        if agents[i].timer1 > 0
            agents[i].timer1 -= 1
            # start infectious period
            if (agents[i].timer1 == 0)
                agents[i].state = :I
                agents[i].timer2 = parameters[:infectious_period]
            end
        elseif agents[i].timer2 > 0
            agents[i].timer2 -= 1
            # check recovery
            if (agents[i].timer2 == 0)
                recover!(agents[i])
            end
        end
    end
end

end

################################################

# SIR
# # initialize an Nx1 matrix of SIR agents (Julia is column-major)
agents = Matrix{ABM.SIRAgent}(undef, ABM.parameters[:num_agents], ABM.parameters[:num_spatial_units])
for i in eachindex(agents)
    agents[i] = ABM.SIRAgent()
end
# seed infections
ABM.infect!(agents[1])

for t = 1:ABM.parameters[:num_timesteps]
    ABM.transmit!(agents)
    ABM.step!(agents)
end

println("SIR ENDSTATE:")
println(count(x -> x.state == :S, agents))
println(count(x -> x.state == :I, agents))
println(count(x -> x.state == :R, agents))


################################################
# SEIR
# # initialize an Nx1 matrix of SIR agents (Julia is column-major)
agents = Matrix{ABM.SEIRAgent}(undef, ABM.parameters[:num_agents], ABM.parameters[:num_spatial_units])
for i in eachindex(agents)
    agents[i] = ABM.SEIRAgent()
end
# seed infections
ABM.infect!(agents[1])

for t = 1:ABM.parameters[:num_timesteps]
    ABM.transmit!(agents)
    ABM.step!(agents)
end

println("SEIR ENDSTATE:")
println(count(x -> x.state == :S, agents))
println(count(x -> x.state == :E, agents))
println(count(x -> x.state == :I, agents))
println(count(x -> x.state == :R, agents))