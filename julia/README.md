README
======

Toy models using Julia. Right now the parameters and method are meant to follow Chistopher Lorton's `well-mixed-abc` branch.

# Getting started

## Examples

Example scripts are in the form `example_XXX.jl` and can be run:

1. From the REPL:
```
julia --project=.
julia> include("example_sir.jl")
``````
2. Or from the command line:
``` julia --project=. example_sir.jl```

Right now the models only take R0 and popsize as arguments. Everything else is hard coded in the repository.

To use multiple threads you can call initialize julia with the `-threads N` flag:
``` julia --project=. --threads 4 example_sir.jl```

## Benchmarks

- `benchmark_num_agents.jl`: Benchmarking memory and runtime vs number of agents.

# Installing Julia

To install Julia you can use the [juliup](https://github.com/JuliaLang/juliaup) version manager.

For Mac and Linux:
```
curl -fsSL https://install.julialang.org | sh
```

