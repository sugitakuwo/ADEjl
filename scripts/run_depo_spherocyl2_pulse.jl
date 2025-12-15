#!/usr/bin/env julia
using Pkg
project_root = joinpath(@__DIR__, "..")
Pkg.activate(project_root)
Pkg.instantiate()

using ADE_MOL

"""
Run deposition fitting for spherocylinder blocking.
Pass `--optimize` to enable parameter optimization.
"""
function main()
    go_optim = any(arg -> arg == "--optimize", ARGS)
    fit_ADEdepo_spherocyl2_pulse(; go_optim)
end

main()
