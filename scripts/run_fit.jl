#!/usr/bin/env julia
using Pkg
project_root = joinpath(@__DIR__, "..")
Pkg.activate(project_root)
Pkg.develop(path=project_root)
Pkg.instantiate()

using ADE_MOL

"""
Run tracer fitting with optional optimization step.
Pass `--optimize` to enable parameter optimization.
"""
function main()
    go_optim = any(arg -> arg == "--optimize", ARGS)
    fit_ADEsolute_pulse(; go_optim)
end

main()
