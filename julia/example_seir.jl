# Basic SEIR
using CSV
include("ToyModels.jl")
import .Models

df = @time Models.SEIR(Int(1e4), 2.5)
CSV.write("SEIR.csv", df)