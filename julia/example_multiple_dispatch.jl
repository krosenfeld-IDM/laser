include("MDModels.jl")
import .ABM

p = ABM.SEIRParameters(100_000, 0 , num_timesteps=365)
agents = ABM.init_model(p)
S = ABM.run_model!(agents, p)
print(S / p.num_agents)