# Basic SIR
using CSV
include("ToyModels.jl")
import .Models

df = @time Models.SIR(Int(1e6), 2.5)
CSV.write("SIR.csv", df)