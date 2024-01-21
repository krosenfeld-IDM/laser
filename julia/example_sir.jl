# Basic SIR
cd(@__DIR__)
using CSV
using BenchmarkTools
include("ToyModels.jl")
import .Models

df = @btime Models.SIR(Int(1e7), 2.5) samples=3
# CSV.write("SIR.csv", df)