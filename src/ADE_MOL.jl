module ADE_MOL

using Dates
using DelimitedFiles
using LinearAlgebra
using OrdinaryDiffEq
using Optimization
using OptimizationOptimJL
using Optim
using Plots
using Printf
using QuadGK
using RecipesBase
using SpecialFunctions
using Statistics

# Plotting recipe to visualize concentration profiles over time.
@recipe function plot(range::StepRangeLen, sol::ODESolution)
    ts = sol.t
    xs = collect(range[2:end-1])
    t_out = collect(0:0.2:2)
    k = 1

    for i in eachindex(ts)
        if ts[i] ≥ t_out[k]
            @series begin
                ylabel --> "c"
                xlabel --> "x"
                ylims --> (0, 1)
                label --> @sprintf "t = %1.1f" ts[i]
                xs, sol.u[i][1:length(xs)]
            end
            k += 1
            if k > length(t_out)
                break
            end
        end
    end
    xlims --> (0, 1)
end

include("utils.jl")
include("pulse_inputs.jl")
include("blocking.jl")
include("ade_solute.jl")
include("ade_deposition.jl")
include("fit_solute.jl")

using .ADEdeposition
export fit_ADEsolute_pulse, fit_ADEdepo_pulse, fit_ADEdepo_spherocyl2_pulse, ADE_depo_RSA_nonSP_pulse,
    load_depo_config

end
