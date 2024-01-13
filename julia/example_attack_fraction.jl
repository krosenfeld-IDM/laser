"""
Attack fraction test
"""

using Plots
using DataFrames
using CSV
using SpecialFunctions
using Roots

include("ToyModels.jl")
import .Models


# analytic solution
function KMlimit(x, R0)
    1 .- x .- exp(-x.*R0)
end


# calculate attack fraction across a sweep of R0 values
function generate_samples(model, num_samples::Int=100)
    Zs = zeros(num_samples)
    R0s = zeros(num_samples)
    popsize = 100_000 # number of agents
    for (i, r0) in enumerate(range(0.5, 1.75, length=num_samples))
        df = model(popsize, r0)
        Zs[i] = 1- df.S[end] / sum(df[1,:])
        R0s[i] = r0
    end
    return (Zs, R0s)
end


# main function to run the model (optionally) and generate the figure
function main(model)

    # run the model (or not)
    write_samples = true
    if write_samples
        (Zs, R0s) = generate_samples(model, 500)
        df = DataFrame(Z=Zs, R0=R0s)
        CSV.write("attack_fraction.csv", df)
    else
        df = DataFrame(CSV.File("attack_fraction.csv"))
    end

    # make figure of data
    p = plot(df.R0, df.Z, seriestype=:scatter, label="Simulation")

    # add analytic solution
    KM = zeros(length(df.R0))
    R0 = range(1.0, 2.0, length=50)
    for (i, R0) in enumerate(R0)
        KM[i] = find_zero(x -> KMlimit(x, R0), 0.5)
    end
    plot!(R0, KM, seriestype=:line, label="Analytic")

    xlabel!(p, "R0")
    ylabel!(p, "Attack fraction")
end


# script call (this is where you choose which model to run)
main(Models.SEIR)