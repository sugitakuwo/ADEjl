#!/usr/bin/env julia
using Pkg
project_root = joinpath(@__DIR__, "..")
Pkg.activate(project_root)
Pkg.develop(path=project_root)
Pkg.instantiate()

using ADE_MOL
using ADE_MOL.Blocking: B_RSA_SP2, B_RSA_nonSP_spherocyl2, B_RSA_nonSP_gamma2, B_none

"""
Run deposition fitting with optional optimization step.
Pass `--optimize` to enable parameter optimization.
Optional flags:
- `--config <path>`: TOML config overrides.
- `--shape <legacy|sphere|spherocyl|rod>` or `--rod`.
- `--blocking <sp2|spherocyl2|gamma2|none>`.
- `--fit-theta-mx` and `--theta-mx <value>`.
- `--fit-kdesorp` and `--kdesorp <value>`.
- `--compare-blocking`, `--scan`, `--plot-scan`.
- `--use-tracer` or `--tracer-fit <path>`.
"""
function parse_args(args)
    opts = Dict{String, Any}()
    i = 1
    while i <= length(args)
        arg = args[i]
        if arg == "--optimize"
            opts["optimize"] = true
            i += 1
        elseif arg == "--config"
            i += 1
            i > length(args) && error("Missing value for --config")
            opts["config_path"] = args[i]
            i += 1
        elseif arg == "--shape"
            i += 1
            i > length(args) && error("Missing value for --shape")
            opts["shape"] = lowercase(args[i])
            i += 1
        elseif arg == "--rod"
            opts["shape"] = "rod"
            i += 1
        elseif arg == "--blocking"
            i += 1
            i > length(args) && error("Missing value for --blocking")
            opts["blocking"] = lowercase(args[i])
            i += 1
        elseif arg == "--fit-theta-mx"
            opts["fit_theta_mx"] = true
            i += 1
        elseif arg == "--theta-mx"
            i += 1
            i > length(args) && error("Missing value for --theta-mx")
            opts["theta_mx"] = parse(Float64, args[i])
            i += 1
        elseif arg == "--fit-kdesorp"
            opts["fit_kdesorp"] = true
            i += 1
        elseif arg == "--kdesorp"
            i += 1
            i > length(args) && error("Missing value for --kdesorp")
            opts["kdesorp"] = parse(Float64, args[i])
            i += 1
        elseif arg == "--compare-blocking"
            opts["compare_blocking"] = true
            i += 1
        elseif arg == "--scan"
            opts["scan_pairs"] = true
            i += 1
        elseif arg == "--plot-scan"
            opts["plot_scan"] = true
            i += 1
        elseif arg == "--use-tracer"
            opts["use_tracer_fit"] = true
            i += 1
        elseif arg == "--tracer-fit"
            i += 1
            i > length(args) && error("Missing value for --tracer-fit")
            opts["tracer_fit_path"] = args[i]
            i += 1
        elseif arg == "--particle-length-nm"
            i += 1
            i > length(args) && error("Missing value for --particle-length-nm")
            opts["particle_length_nm"] = parse(Float64, args[i])
            i += 1
        elseif arg == "--particle-diameter-nm"
            i += 1
            i > length(args) && error("Missing value for --particle-diameter-nm")
            opts["particle_diameter_nm"] = parse(Float64, args[i])
            i += 1
        else
            error("Unknown argument: $(arg)")
        end
    end
    return opts
end

function resolve_blocking(name)
    name === nothing && return nothing
    if name in ("sp2", "sphere", "sp")
        return B_RSA_SP2
    elseif name in ("spherocyl2", "spherocyl", "spherocylinder")
        return B_RSA_nonSP_spherocyl2
    elseif name in ("gamma2", "rod", "stadium")
        return B_RSA_nonSP_gamma2
    elseif name in ("none", "off")
        return B_none
    else
        error("Unknown blocking option: $(name)")
    end
end

function merge_section!(config, section, overrides)
    isempty(overrides) && return config
    base = get!(config, section, Dict{String, Any}())
    for (k, v) in overrides
        base[k] = v
    end
    return config
end

function main()
    opts = parse_args(ARGS)
    go_optim = get(opts, "optimize", false)
    config_path = get(opts, "config_path", nothing)

    config = config_path === nothing ? Dict{String, Any}() : load_depo_config(config_path)

    fit_overrides = Dict{String, Any}()
    if haskey(opts, "theta_mx")
        fit_overrides["theta_mx_fixed"] = opts["theta_mx"]
        fit_overrides["theta_mx_init"] = opts["theta_mx"]
    end
    if haskey(opts, "kdesorp")
        fit_overrides["kdesorp_fixed"] = opts["kdesorp"]
        fit_overrides["kdesorp_init"] = opts["kdesorp"]
    end
    merge_section!(config, "fit", fit_overrides)

    shape_str = get(opts, "shape", nothing)
    particle_shape = shape_str === nothing ? nothing : Symbol(shape_str)

    blocking_choice = resolve_blocking(get(opts, "blocking", nothing))
    if blocking_choice === nothing && particle_shape !== nothing
        if particle_shape == :rod
            blocking_choice = B_RSA_nonSP_gamma2
        elseif particle_shape == :spherocyl
            blocking_choice = B_RSA_nonSP_spherocyl2
        end
    end
    blocking_choice = blocking_choice === nothing ? B_RSA_SP2 : blocking_choice

    fit_ADEdepo_pulse(; go_optim, blocking_fn = blocking_choice, config,
        particle_shape = particle_shape,
        particle_length_nm = get(opts, "particle_length_nm", nothing),
        particle_diameter_nm = get(opts, "particle_diameter_nm", nothing),
        use_tracer_fit = get(opts, "use_tracer_fit", nothing),
        tracer_fit_path = get(opts, "tracer_fit_path", nothing),
        fit_theta_mx = get(opts, "fit_theta_mx", nothing),
        fit_kdesorp = get(opts, "fit_kdesorp", nothing),
        compare_blocking = get(opts, "compare_blocking", nothing),
        scan_pairs = get(opts, "scan_pairs", nothing),
        plot_scan = get(opts, "plot_scan", nothing))
end

main()
