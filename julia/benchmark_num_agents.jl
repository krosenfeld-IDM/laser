"""
Capture time and memory usage as a function of the number of agents.
"""

cd(@__DIR__)
using Plots
using BenchmarkTools
using ProgressBars

include("ToyModels.jl")
import .Models

include("MDModels.jl")
import .ABM

function abm_seir(n, r0)
    p = ABM.SEIRParameters(n, r0, num_timesteps=365)
    agents = ABM.init_model(p)
    ABM.run_model!(agents, p)
    return nothing
end

function run_benchmark()
    # number of agents (in log space)
    ns = [2, 3, 4, 5, 6, 7]

    # list of the SIR and SEIR models contained in Model
    models = [Models.SIR, Models.SEIR, abm_seir]

    # loop over the models list and add their benchmark results to a dict
    results = Dict()
    for model in ProgressBar(models)
        measure_time = zeros(length(ns))
        measure_memory = zeros(length(ns))
        for i in eachindex(ns)
            n = ns[i]
            b = @benchmark $model(Int(10^$n), 2.5) samples=10
            measure_time[i] = median(b.times)
            measure_memory[i] = b.memory
        end
        results[string(model)] = (measure_time, measure_memory)
    end

    return ns, results
end

ns, results = run_benchmark()

# make figure for time
plot(ns, log10.(results["abm_seir"][1]), seriestype=:scatter, label="ABM", dpi=300)
plot!(ns, log10.(results["SIR"][1]), seriestype=:scatter, label="SIR")
plot!(ns, log10.(results["SEIR"][1]), seriestype=:scatter, label="SEIR")
xlabel!("log10(N)")
ylabel!("log10(seconds)")
savefig("benchmark_num_agents_time.png")

# make figure for memory
plot(ns, log10.(results["abm_seir"][2]), seriestype=:scatter, label="ABM", dpi=300)
plot!(ns, log10.(results["SIR"][2]), seriestype=:scatter, label="SIR")
plot!(ns, log10.(results["SEIR"][2]), seriestype=:scatter, label="SEIR")
xlabel!("log10(N)")
ylabel!("log10(memory)")
savefig("benchmark_num_agents_memory.png")