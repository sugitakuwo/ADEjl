module ADEdeposition
    export ADE_depo_RSA_nonSP_pulse, Lsq_ADEdepo_pulse, fit_ADEdepo_pulse, fit_ADEdepo_spherocyl2_pulse,
        load_depo_config

    using DomainSets
    using DataInterpolations
    using Interpolations
    using MethodOfLines
    using ModelingToolkit
    using OrdinaryDiffEq
    using Symbolics
    using Dates
    using DelimitedFiles
    using Optim
    using Plots
    using QuadGK
    using TOML

    using ..ADEsolute: intpsol
    using ..PulseInputs: pulse_input_SP
    using ..Blocking: B_RSA_SP2, B_RSA_nonSP_spherocyl2, B_RSA_nonSP_gamma2, B_none

    "Dispatch-safe evaluation for blocking functions with or without shape parameters."
    function blocking_value(blocking_fn, theta, thmax, shape_param, theta_mx)
        theta_eff = theta / theta_mx
        return applicable(blocking_fn, theta_eff, thmax, shape_param) ?
            blocking_fn(theta_eff, thmax, shape_param) : blocking_fn(theta_eff, thmax)
    end

    "Projected side-on area for a rod (stadium approximation)."
    rod_projection_area(L, d) = L * d + (pi * d^2) / 4

    "Perimeter for a rod projection (stadium approximation)."
    rod_projection_perimeter(L, d) = 2 * L + pi * d

    "Shape factor gamma_p from stadium perimeter and area."
    rod_gamma_stadium(L, d) = rod_projection_perimeter(L, d)^2 / (4 * pi * rod_projection_area(L, d))

    "Derive particle geometry parameters needed for blocking/coverage."
    function particle_geometry(; shape = :legacy, length = 87e-9, diameter = 7.3e-9)
        ap = diameter / 2
        if shape == :legacy
            bp = length / 2
            As = bp / ap
            area_prolate = pi * bp * ap
            area_ave = 0.5 * (area_prolate + pi * ap^2)
            Areap = area_ave
        elseif shape == :rod
            Areap = rod_projection_area(length, diameter)
            As = length / diameter
        elseif shape == :spherocyl
            Areap = rod_projection_area(length, diameter)
            As = length / diameter
        elseif shape == :sphere
            Areap = pi * ap^2
            As = 1.0
        else
            error("Unsupported particle shape: $(shape)")
        end
        gamma_p = rod_gamma_stadium(length, diameter)
        return (; Areap, As, gamma_p, length, diameter, shape)
    end

    "Select the shape parameter matching the blocking function."
    function select_shape_param(blocking_fn, geom)
        if blocking_fn === B_RSA_nonSP_gamma2
            return geom.gamma_p, "gamma_p"
        elseif blocking_fn === B_RSA_nonSP_spherocyl2
            return geom.As, "As"
        else
            return geom.As, "As"
        end
    end

    "Interpolate a surface coverage profile for reuse across injections."
    function theta_intp(x, t, sol_in, dx)
        A_x1 = 0.0:dx:1.0
        A = sol_in(t, A_x1)[2]
        itp = interpolate(A, BSpline(Cubic(Line(OnGrid()))))
        sitp1 = scale(itp, A_x1)
        return sitp1(x)
    end

    @register_symbolic theta_intp(x, t, sol_in, dx)

    "Solve ADE with deposition and RSA blocking for a pulse injection."
    function ADE_depo_RSA_nonSP_pulse(; L = 5.3e-2, xmax = 1.0, tmax = 2.5, vp_in = 5.0e-4, D_in = 0.01,
        f_in = 1.0e-2, kdepo_in = 1.0e-2, Areap = 1.0e-5, C0_in = 6.58e19, theta_max = 1.0,
        shape_param = 10.0, theta_mx_in = 1.0, kdesorp_in = 0.0, dx = 0.01, pulse_width = 0.2,
        delta = 0.02, theta_init = nothing, x_init = nothing, solver = Tsit5(), blocking_fn = B_RSA_SP2)
        @parameters t x
        @parameters vp, Lc, f, C0, Dsol, kdepo, Ap, thmax, shape_p, theta_mx, kdesorp
        @variables c(..), theta(..)

        Dt = Differential(t)
        Dx = Differential(x)
        Dxx = Differential(x)^2

        block = blocking_value(blocking_fn, theta(x, t), thmax, shape_p, theta_mx)
        eqs = [
            Dt(c(x, t)) ~ Dsol / (vp * Lc) * Dxx(c(x, t)) - Dx(c(x, t)) -
            Lc / vp * f * (kdepo * block * c(x, t) - kdesorp * theta(x, t) / (Ap * C0)),
            Dt(theta(x, t)) ~ Lc / vp * (kdepo * block * Ap * C0 * c(x, t) - kdesorp * theta(x, t)),
        ]

        bcs = if theta_init !== nothing && x_init !== nothing
            [
                c(x, 0.0) ~ 0.0,
                c(0.0, t) ~ pulse_input_SP(t, pulse_width, delta),
                Dx(c(1, t)) ~ 0.0,
                theta(x, 0) ~ intpsol(x, x_init, theta_init[:, end]),
            ]
        else
            [
                c(x, 0.0) ~ 0.0,
                c(0.0, t) ~ pulse_input_SP(t, pulse_width, delta),
                Dx(c(1, t)) ~ 0.0,
                theta(x, 0) ~ 0.0,
            ]
        end

        domains = [x ∈ Interval(0.0, xmax), t ∈ Interval(0.0, tmax)]

        @named pdesys = PDESystem(eqs, bcs, domains, [x, t], [c(x, t), theta(x, t)],
            [vp => vp_in, Dsol => D_in, Lc => L, f => f_in, kdepo => kdepo_in, Ap => Areap,
                shape_p => shape_param, theta_mx => theta_mx_in, kdesorp => kdesorp_in, C0 => C0_in,
                thmax => theta_max])

        disc = MOLFiniteDifference([x => dx], t, approx_order = 6)
        prob = discretize(pdesys, disc)

        sol = solve(prob, solver)
        discrete_x = sol[x]
        discrete_t = sol[t]
        solc = sol[c(x, t)]
        soltheta = sol[theta(x, t)]

        return discrete_x, discrete_t, solc, soltheta, sol
    end

    "Load and scale experimental data for deposition pulses."
    function load_experiments(; pattern = "CN", Absinput, tatmax, tp, tmax = 2.5)
        files = sort(filter(f -> occursin(pattern, f), readdir()))
        length(files) < 3 && error("Need at least three experimental files matching pattern=$(pattern)")
        data = [readdlm(files[i], ',', Float64, skipstart = 2) for i in 1:3]

        for arr in data
            arr[:, 2] ./= Absinput
            arr[:, 1] = (arr[:, 1] .- (tatmax - tp)) ./ tp
        end

        return [arr[0.0 .< arr[:, 1] .< tmax, :] for arr in data]
    end

    "Load deposition configuration from a TOML file."
    load_depo_config(path::AbstractString) = TOML.parsefile(path)

    "Helper to access a config section."
    config_section(config, key) = config === nothing ? Dict{String, Any}() : get(config, key, Dict{String, Any}())

    "Helper to fetch numeric config values with defaults."
    function config_get(section, key, default)
        if haskey(section, key)
            val = section[key]
            return val isa Integer ? Float64(val) : val
        end
        return default
    end

    "Helper to fetch boolean config values with defaults."
    function config_get_bool(section, key, default)
        return haskey(section, key) ? Bool(section[key]) : default
    end

    "Helper to fetch string config values with defaults."
    function config_get_string(section, key, default)
        return haskey(section, key) ? String(section[key]) : default
    end

    "Find the latest tracer fit output file."
    function latest_tracer_fit_file(; pattern = "Out_optimized_disp_leng_fporo_")
        files = filter(f -> occursin(pattern, f), readdir())
        isempty(files) && return nothing
        return sort(files; by = f -> stat(f).mtime, rev = true)[1]
    end

    "Load tracer fit results from file."
    function load_tracer_fit(path)
        vals = vec(readdlm(path, Float64))
        length(vals) < 2 && error("Tracer fit file must contain alphaL and fporo values.")
        return vals[1], vals[2]
    end

    "Compute transport parameters for deposition."
    function transport_params(params; D_override = nothing)
        vp = params.vapp / params.fporo
        tp = params.L / vp
        D = D_override === nothing ? params.alphaL * vp : D_override
        f = 3.0 * (1.0 - params.fporo) / (params.ac * params.fporo)
        pulse_width = params.tpulse / tp
        return (; vp, tp, D, f, pulse_width)
    end

    "Simulate three consecutive deposition pulses."
    function simulate_depo_pulses(params; kdepo, thmax, theta_mx = 1.0, kdesorp = 0.0,
        blocking_fn = B_RSA_SP2, D_override = nothing)
        trans = transport_params(params; D_override)
        xs1, ts1, solc1, soltheta1, sol1 = ADE_depo_RSA_nonSP_pulse(
            L = params.L, xmax = params.xmax, tmax = params.tmax, vp_in = trans.vp, D_in = trans.D,
            f_in = trans.f, kdepo_in = kdepo, Areap = params.Areap, C0_in = params.C0,
            theta_max = thmax, shape_param = params.shape_param, theta_mx_in = theta_mx,
            kdesorp_in = kdesorp, pulse_width = trans.pulse_width, dx = params.dx, solver = params.solver,
            blocking_fn = blocking_fn
        )
        xs2, ts2, solc2, soltheta2, sol2 = ADE_depo_RSA_nonSP_pulse(
            L = params.L, xmax = params.xmax, tmax = params.tmax, vp_in = trans.vp, D_in = trans.D,
            f_in = trans.f, kdepo_in = kdepo, Areap = params.Areap, C0_in = params.C0,
            theta_max = thmax, shape_param = params.shape_param, theta_mx_in = theta_mx,
            kdesorp_in = kdesorp, pulse_width = trans.pulse_width, dx = params.dx,
            theta_init = soltheta1, x_init = xs1, solver = params.solver, blocking_fn = blocking_fn
        )
        xs3, ts3, solc3, soltheta3, sol3 = ADE_depo_RSA_nonSP_pulse(
            L = params.L, xmax = params.xmax, tmax = params.tmax, vp_in = trans.vp, D_in = trans.D,
            f_in = trans.f, kdepo_in = kdepo, Areap = params.Areap, C0_in = params.C0,
            theta_max = thmax, shape_param = params.shape_param, theta_mx_in = theta_mx,
            kdesorp_in = kdesorp, pulse_width = trans.pulse_width, dx = params.dx,
            theta_init = soltheta2, x_init = xs2, solver = params.solver, blocking_fn = blocking_fn
        )
        return (; xs1, ts1, solc1, soltheta1, sol1,
            xs2, ts2, solc2, soltheta2, sol2,
            xs3, ts3, solc3, soltheta3, sol3, trans)
    end

    "Least squares for a single dataset and solution."
    function lsq_dataset(exp_data, sol)
        lsqsum = 0.0
        for ii in axes(exp_data, 1)
            solc_val = sol(exp_data[ii, 1], 1.0)
            lsqsum += (exp_data[ii, 2] - solc_val[1])^2
        end
        return lsqsum
    end

    "Total least squares across three datasets."
    function total_lsq(datasets, sols)
        return lsq_dataset(datasets[1], sols.sol1) +
               lsq_dataset(datasets[2], sols.sol2) +
               lsq_dataset(datasets[3], sols.sol3)
    end

    "Unpack optimization parameters based on fit spec."
    function unpack_fit_params(p::Vector, fit_spec)
        idx = 1
        kdepo = p[idx]; idx += 1
        thmax = fit_spec.fit_thmax ? p[idx] : fit_spec.thmax_fixed
        idx += fit_spec.fit_thmax ? 1 : 0
        theta_mx = fit_spec.fit_theta_mx ? p[idx] : fit_spec.theta_mx_fixed
        idx += fit_spec.fit_theta_mx ? 1 : 0
        kdesorp = fit_spec.fit_kdesorp ? p[idx] : fit_spec.kdesorp_fixed
        return kdepo, thmax, theta_mx, kdesorp
    end

    "Count number of free fit parameters."
    fit_param_count(fit_spec) = 1 +
        (fit_spec.fit_thmax ? 1 : 0) +
        (fit_spec.fit_theta_mx ? 1 : 0) +
        (fit_spec.fit_kdesorp ? 1 : 0)

    "AIC/BIC from sum of squares."
    function aic_bic(sse, n_obs, n_params)
        if sse <= 0 || n_obs <= 0
            return (; aic = NaN, bic = NaN)
        end
        aic = n_obs * log(sse / n_obs) + 2 * n_params
        bic = n_obs * log(sse / n_obs) + n_params * log(n_obs)
        return (; aic, bic)
    end

    "Write a grid scan result to file."
    function write_scan_csv(outfile, xvals, yvals, sse; x_label, y_label)
        open(outfile, "w") do f_io
            println(f_io, "# Grid scan: $(x_label) vs $(y_label) with SSE values")
            print(f_io, x_label)
            for y in yvals
                print(f_io, " ", y)
            end
            println(f_io)
            for (i, x) in enumerate(xvals)
                print(f_io, x)
                for j in eachindex(yvals)
                    print(f_io, " ", sse[i, j])
                end
                println(f_io)
            end
        end
    end

    "Write a coverage profile to file."
    function write_theta_profile(outfile, xs, theta)
        open(outfile, "w") do f_io
            println(f_io, "#x theta")
            for (xval, tval) in zip(xs, theta)
                println(f_io, xval, " ", tval)
            end
        end
    end

    "Write outlet coverage time series to file."
    function write_theta_outlet(outfile, ts, theta_outlet)
        open(outfile, "w") do f_io
            println(f_io, "#tpv theta_outlet")
            for (tval, th) in zip(ts, theta_outlet)
                println(f_io, tval, " ", th)
            end
        end
    end

    "Write coverage profiles for three pulses."
    function write_coverage_profiles(sols; prefix = "Out")
        write_theta_profile("$(prefix)_theta_profile_pulse1.txt", sols.xs1, sols.soltheta1[:, end])
        write_theta_profile("$(prefix)_theta_profile_pulse2.txt", sols.xs2, sols.soltheta2[:, end])
        write_theta_profile("$(prefix)_theta_profile_pulse3.txt", sols.xs3, sols.soltheta3[:, end])
        write_theta_outlet("$(prefix)_theta_outlet_pulse1.txt", sols.ts1, sols.soltheta1[end, :])
        write_theta_outlet("$(prefix)_theta_outlet_pulse2.txt", sols.ts2, sols.soltheta2[end, :])
        write_theta_outlet("$(prefix)_theta_outlet_pulse3.txt", sols.ts3, sols.soltheta3[end, :])
    end

    "Plot BTCs and coverage profiles."
    function plot_depo_profiles(sols; outfile = "Out_depo_profiles.png")
        p1 = plot(sols.ts1, sols.solc1[end, :], xlabel = "tpv", ylabel = "c/c0",
            label = "1st", title = "BTC (outlet)")
        plot!(p1, sols.ts2, sols.solc2[end, :], label = "2nd")
        plot!(p1, sols.ts3, sols.solc3[end, :], label = "3rd")

        p2 = plot(sols.xs1, sols.soltheta1[:, end], xlabel = "x", ylabel = "theta",
            label = "1st", title = "Coverage (end of pulse)")
        plot!(p2, sols.xs2, sols.soltheta2[:, end], label = "2nd")
        plot!(p2, sols.xs3, sols.soltheta3[:, end], label = "3rd")

        plot(p1, p2, layout = (1, 2), size = (1500, 600))
        savefig(outfile)
    end

    "Resolve blocking model names to functions."
    function resolve_blocking_models(names::Vector{String})
        mapping = Dict(
            "sp2" => B_RSA_SP2,
            "sphere" => B_RSA_SP2,
            "spherocyl2" => B_RSA_nonSP_spherocyl2,
            "spherocyl" => B_RSA_nonSP_spherocyl2,
            "gamma2" => B_RSA_nonSP_gamma2,
            "rod" => B_RSA_nonSP_gamma2,
            "none" => B_none,
            "off" => B_none
        )
        models = Vector{Tuple{String, Function}}()
        for name in names
            haskey(mapping, name) || error("Unknown blocking model: $(name)")
            push!(models, (name, mapping[name]))
        end
        return models
    end

    "Plot ASF curves for multiple blocking models."
    function plot_blocking_models(models, geom; thmax, theta_mx = 1.0, outfile = "Out_blocking_compare.png")
        p = plot(xlabel = "theta", ylabel = "ASF", title = "Blocking comparison")
        theta = range(0.0, stop = thmax * theta_mx, length = 200)
        for (name, fn) in models
            shape_param, _ = select_shape_param(fn, geom)
            vals = [blocking_value(fn, th, thmax, shape_param, theta_mx) for th in theta]
            plot!(p, theta, vals, label = name)
        end
        savefig(p, outfile)
    end

    "Write ASF curves for multiple blocking models."
    function write_blocking_models(models, geom; thmax, theta_mx = 1.0, outfile = "Out_blocking_compare.txt")
        theta = range(0.0, stop = thmax * theta_mx, length = 200)
        open(outfile, "w") do f_io
            println(f_io, "#theta " * join(first.(models), " "))
            for th in theta
                vals = Vector{Float64}()
                for (_, fn) in models
                    shape_param, _ = select_shape_param(fn, geom)
                    push!(vals, blocking_value(fn, th, thmax, shape_param, theta_mx))
                end
                println(f_io, th, " ", join(vals, " "))
            end
        end
    end

    "Least squares objective for fitting deposition parameters."
    function Lsq_ADEdepo_pulse(p::Vector, datasets, params, fit_spec; blocking_fn = B_RSA_SP2)
        kdepo, thmax, theta_mx, kdesorp = unpack_fit_params(p, fit_spec)
        sols = simulate_depo_pulses(params; kdepo, thmax, theta_mx, kdesorp, blocking_fn)
        return total_lsq(datasets, sols)
    end

    "Fit deposition parameters and simulate three consecutive injections."
    function fit_ADEdepo_pulse(; go_optim = false, blocking_fn = B_RSA_SP2, config_path = nothing, config = nothing,
        particle_shape::Union{Nothing, Symbol} = nothing, particle_length_nm = nothing, particle_diameter_nm = nothing,
        use_tracer_fit::Union{Nothing, Bool} = nothing, tracer_fit_path = nothing,
        fit_thmax::Union{Nothing, Bool} = nothing, fit_theta_mx::Union{Nothing, Bool} = nothing,
        fit_kdesorp::Union{Nothing, Bool} = nothing, compare_blocking::Union{Nothing, Bool} = nothing,
        scan_pairs::Union{Nothing, Bool} = nothing, plot_scan::Union{Nothing, Bool} = nothing,
        write_profiles::Union{Nothing, Bool} = nothing, plot_profiles::Union{Nothing, Bool} = nothing,
        compare_blocking_models::Union{Nothing, Bool} = nothing)
        Absinput = 0.3468
        tatmax = 141.0
        tpulse = 20.0

        L = 0.053
        xmax = 1.0
        tmax = 2.5
        dc = 0.03e-2
        vapp = 0.02e-2
        fporo = 0.383
        alphaL = 0.0007
        dx = 0.005
        solver = RadauIIA5()

        particle_length = 87e-9
        particle_diameter = 7.3e-9
        shape_default = :legacy

        C0 = 2 * 9.15e19

        kdepo_init = 5.0e-7
        kdepo_lower = 1.0e-7
        kdepo_upper = 2.0e-6
        kdepo_fixed = 8.95e-7
        thmax_init = 0.18
        thmax_lower = 0.13
        thmax_upper = 0.25
        thmax_fixed = 0.193
        theta_mx_init = 1.0
        theta_mx_lower = 0.2
        theta_mx_upper = 1.0
        theta_mx_fixed = 1.0
        kdesorp_init = 0.0
        kdesorp_lower = 0.0
        kdesorp_upper = 1.0e-4
        kdesorp_fixed = 0.0
        n_particles = 5
        n_iterations = 10

        config = config === nothing && config_path !== nothing ? load_depo_config(config_path) : config

        exp_cfg = config_section(config, "experiment")
        Absinput = config_get(exp_cfg, "Absinput", Absinput)
        tatmax = config_get(exp_cfg, "tatmax", tatmax)
        tpulse = config_get(exp_cfg, "tpulse", tpulse)
        tmax = config_get(exp_cfg, "tmax", tmax)

        column_cfg = config_section(config, "column")
        L = config_get(column_cfg, "L", L)
        xmax = config_get(column_cfg, "xmax", xmax)
        tmax = config_get(column_cfg, "tmax", tmax)
        dc = config_get(column_cfg, "dc", dc)
        vapp = config_get(column_cfg, "vapp", vapp)
        fporo = config_get(column_cfg, "fporo", fporo)
        alphaL = config_get(column_cfg, "alphaL", alphaL)
        dx = config_get(column_cfg, "dx", dx)

        particle_cfg = config_section(config, "particle")
        C0 = config_get(particle_cfg, "C0", C0)
        shape_default = Symbol(config_get_string(particle_cfg, "shape", string(shape_default)))

        length_nm = config_get(particle_cfg, "length_nm", nothing)
        diameter_nm = config_get(particle_cfg, "diameter_nm", nothing)
        length_m = config_get(particle_cfg, "length_m", nothing)
        diameter_m = config_get(particle_cfg, "diameter_m", nothing)
        if length_nm !== nothing
            particle_length = length_nm * 1.0e-9
        elseif length_m !== nothing
            particle_length = length_m
        end
        if diameter_nm !== nothing
            particle_diameter = diameter_nm * 1.0e-9
        elseif diameter_m !== nothing
            particle_diameter = diameter_m
        end

        fit_cfg = config_section(config, "fit")
        kdepo_init = config_get(fit_cfg, "kdepo_init", kdepo_init)
        kdepo_lower = config_get(fit_cfg, "kdepo_lower", kdepo_lower)
        kdepo_upper = config_get(fit_cfg, "kdepo_upper", kdepo_upper)
        kdepo_fixed = config_get(fit_cfg, "kdepo_fixed", kdepo_fixed)
        thmax_init = config_get(fit_cfg, "thmax_init", thmax_init)
        thmax_lower = config_get(fit_cfg, "thmax_lower", thmax_lower)
        thmax_upper = config_get(fit_cfg, "thmax_upper", thmax_upper)
        thmax_fixed = config_get(fit_cfg, "thmax_fixed", thmax_fixed)
        theta_mx_init = config_get(fit_cfg, "theta_mx_init", theta_mx_init)
        theta_mx_lower = config_get(fit_cfg, "theta_mx_lower", theta_mx_lower)
        theta_mx_upper = config_get(fit_cfg, "theta_mx_upper", theta_mx_upper)
        theta_mx_fixed = config_get(fit_cfg, "theta_mx_fixed", theta_mx_fixed)
        kdesorp_init = config_get(fit_cfg, "kdesorp_init", kdesorp_init)
        kdesorp_lower = config_get(fit_cfg, "kdesorp_lower", kdesorp_lower)
        kdesorp_upper = config_get(fit_cfg, "kdesorp_upper", kdesorp_upper)
        kdesorp_fixed = config_get(fit_cfg, "kdesorp_fixed", kdesorp_fixed)
        n_particles = Int(config_get(fit_cfg, "n_particles", n_particles))
        n_iterations = Int(config_get(fit_cfg, "iterations", n_iterations))

        fit_thmax = fit_thmax === nothing ? config_get_bool(fit_cfg, "fit_thmax", true) : fit_thmax
        fit_theta_mx = fit_theta_mx === nothing ? config_get_bool(fit_cfg, "fit_theta_mx", false) : fit_theta_mx
        fit_kdesorp = fit_kdesorp === nothing ? config_get_bool(fit_cfg, "fit_kdesorp", false) : fit_kdesorp

        tracer_cfg = config_section(config, "tracer")
        use_tracer_fit = use_tracer_fit === nothing ? config_get_bool(tracer_cfg, "use_tracer_fit", false) : use_tracer_fit
        tracer_fit_path = tracer_fit_path === nothing ? config_get(tracer_cfg, "fit_path", nothing) : tracer_fit_path

        analysis_cfg = config_section(config, "analysis")
        compare_blocking = compare_blocking === nothing ? config_get_bool(analysis_cfg, "compare_blocking", false) : compare_blocking
        scan_pairs = scan_pairs === nothing ? config_get_bool(analysis_cfg, "scan_pairs", false) : scan_pairs
        plot_scan = plot_scan === nothing ? config_get_bool(analysis_cfg, "plot_scan", false) : plot_scan
        write_profiles = write_profiles === nothing ? config_get_bool(analysis_cfg, "write_profiles", false) : write_profiles
        plot_profiles = plot_profiles === nothing ? config_get_bool(analysis_cfg, "plot_profiles", false) : plot_profiles
        compare_blocking_models = compare_blocking_models === nothing ?
            config_get_bool(analysis_cfg, "compare_blocking_models", false) : compare_blocking_models
        scan_points = Int(config_get(analysis_cfg, "scan_points", 9))
        scan_factor = config_get(analysis_cfg, "scan_factor", 0.5)
        blocking_model_names = config_get(analysis_cfg, "blocking_models", ["sp2", "spherocyl2", "gamma2", "none"])

        if particle_shape !== nothing
            shape_default = particle_shape
        end
        if particle_length_nm !== nothing
            particle_length = particle_length_nm * 1.0e-9
        end
        if particle_diameter_nm !== nothing
            particle_diameter = particle_diameter_nm * 1.0e-9
        end

        if fit_theta_mx && (theta_mx_lower <= 0 || theta_mx_upper > 1.0)
            error("theta_mx bounds must satisfy 0 < lower and upper <= 1.")
        end
        if kdesorp_lower < 0 || kdesorp_upper < 0
            error("kdesorp bounds must be non-negative.")
        end

        if use_tracer_fit
            tracer_file = tracer_fit_path === nothing ? latest_tracer_fit_file() : tracer_fit_path
            tracer_file === nothing && error("Tracer fit file not found.")
            alphaL, fporo = load_tracer_fit(tracer_file)
        end

        geom = particle_geometry(; shape = shape_default, length = particle_length, diameter = particle_diameter)
        shape_param, shape_label = select_shape_param(blocking_fn, geom)

        ac = 0.5 * dc
        params = (; L, xmax, tmax, vapp, fporo, alphaL, ac, tpulse, dx, solver, Areap = geom.Areap,
            C0, shape_param, dc, shape_label, geom)

        vp = vapp / fporo
        tp = L / vp

        datasets = load_experiments(; Absinput, tatmax, tp, tmax)
        exp1, exp2, exp3 = datasets

        fit_spec = (; fit_thmax, fit_theta_mx, fit_kdesorp,
            thmax_fixed, theta_mx_fixed, kdesorp_fixed)

        initial_x = [kdepo_init]
        lower = [kdepo_lower]
        upper = [kdepo_upper]
        if fit_thmax
            push!(initial_x, thmax_init)
            push!(lower, thmax_lower)
            push!(upper, thmax_upper)
        end
        if fit_theta_mx
            push!(initial_x, theta_mx_init)
            push!(lower, theta_mx_lower)
            push!(upper, theta_mx_upper)
        end
        if fit_kdesorp
            push!(initial_x, kdesorp_init)
            push!(lower, kdesorp_lower)
            push!(upper, kdesorp_upper)
        end

        options = Optim.Options(iterations = n_iterations)

        if go_optim
            res = Optim.optimize(p -> Lsq_ADEdepo_pulse(p, datasets, params, fit_spec; blocking_fn),
                initial_x, ParticleSwarm(lower, upper, n_particles), options)
            kdepo_opt, thmax_opt, theta_mx_opt, kdesorp_opt = unpack_fit_params(Optim.minimizer(res), fit_spec)
        else
            kdepo_opt = kdepo_fixed
            thmax_opt = thmax_fixed
            theta_mx_opt = theta_mx_fixed
            kdesorp_opt = kdesorp_fixed
        end

        sols = simulate_depo_pulses(params; kdepo = kdepo_opt, thmax = thmax_opt,
            theta_mx = theta_mx_opt, kdesorp = kdesorp_opt, blocking_fn = blocking_fn)

        n_obs = sum(size(data, 1) for data in datasets)
        sse = total_lsq(datasets, sols)
        aicbic = aic_bic(sse, n_obs, fit_param_count(fit_spec))

        k_fit = sols.trans.f * kdepo_opt

        cft_cfg = config_section(config, "cft")
        eps_cft = config_get(cft_cfg, "eps", fporo)
        dc_cft = config_get(cft_cfg, "dc", dc)
        v_cft = config_get(cft_cfg, "v", sols.trans.vp)
        if haskey(cft_cfg, "vapp")
            v_cft = config_get(cft_cfg, "vapp", vapp) / eps_cft
        end
        k_cft = (3.0 * (1.0 - eps_cft) / (2.0 * dc_cft)) * v_cft
        eta = k_cft > 0 ? k_fit / k_cft : NaN

        outfile1 = "Out_exp1st_2nd_3rd_rescaled.txt"
        outfile2 = "Out_cal1st_injection.txt"
        outfile3 = "Out_cal2nd_injection.txt"
        outfile4 = "Out_cal3rd_injection.txt"
        outfile5 = "Out_depo_fit_summary.txt"

        open(outfile1, "w") do f_io
            println(f_io, "#Experimental tpv, c/c0 for 1st, 2nd, 3rd injection")
            for ii in axes(exp1, 1)
                println(f_io, exp1[ii, 1], " ", exp1[ii, 2], " ", exp2[ii, 1], " ", exp2[ii, 2], " ", exp3[ii, 1], " ", exp3[ii, 2])
            end
        end

        open(outfile2, "w") do f_io
            println(f_io, "#Calculated tpv, c/c0 for 1st injection")
            for ii in eachindex(sols.ts1)
                println(f_io, sols.ts1[ii], " ", sols.solc1[end, ii])
            end
        end

        open(outfile3, "w") do f_io
            println(f_io, "#Calculated tpv, c/c0 for 2nd injection")
            for ii in eachindex(sols.ts2)
                println(f_io, sols.ts2[ii], " ", sols.solc2[end, ii])
            end
        end

        open(outfile4, "w") do f_io
            println(f_io, "#Calculated tpv, c/c0 for 3rd injection")
            for ii in eachindex(sols.ts3)
                println(f_io, sols.ts3[ii], " ", sols.solc3[end, ii])
            end
        end

        if compare_blocking
            sols_noblock = simulate_depo_pulses(params; kdepo = kdepo_opt, thmax = thmax_opt, theta_mx = theta_mx_opt,
                kdesorp = kdesorp_opt, blocking_fn = B_none)
            sse_noblock = total_lsq(datasets, sols_noblock)
        else
            sse_noblock = NaN
        end

        open(outfile5, "w") do f_io
            println(f_io, "#Deposition fit summary")
            println(f_io, "shape = $(geom.shape)")
            println(f_io, "particle_length_m = $(geom.length)")
            println(f_io, "particle_diameter_m = $(geom.diameter)")
            println(f_io, "Areap = $(geom.Areap)")
            println(f_io, "shape_param_label = $(shape_label)")
            println(f_io, "shape_param = $(shape_param)")
            println(f_io, "kdepo_opt = $(kdepo_opt)")
            println(f_io, "thmax_opt = $(thmax_opt)")
            println(f_io, "theta_mx_opt = $(theta_mx_opt)")
            println(f_io, "kdesorp_opt = $(kdesorp_opt)")
            println(f_io, "k_fit = $(k_fit)")
            println(f_io, "k_cft = $(k_cft)")
            println(f_io, "eta = $(eta)")
            println(f_io, "sse = $(sse)")
            println(f_io, "aic = $(aicbic.aic)")
            println(f_io, "bic = $(aicbic.bic)")
            if compare_blocking
                println(f_io, "sse_noblock = $(sse_noblock)")
            end
        end

        if write_profiles
            write_coverage_profiles(sols)
        end
        if plot_profiles
            plot_depo_profiles(sols)
        end

        if compare_blocking_models
            model_names = String.(blocking_model_names)
            models = resolve_blocking_models(model_names)
            write_blocking_models(models, geom; thmax = thmax_opt, theta_mx = theta_mx_opt)
            plot_blocking_models(models, geom; thmax = thmax_opt, theta_mx = theta_mx_opt)
        end

        if scan_pairs
            kdepo_low = max(kdepo_opt * (1 - scan_factor), kdepo_opt * 0.1)
            kdepo_high = kdepo_opt * (1 + scan_factor)
            d_low = max(sols.trans.D * (1 - scan_factor), sols.trans.D * 0.1)
            d_high = sols.trans.D * (1 + scan_factor)
            theta_low = max(0.05, theta_mx_opt * (1 - scan_factor))
            theta_high = min(1.0, theta_mx_opt * (1 + scan_factor))
            theta_high < theta_low && (theta_high = min(1.0, theta_low + 0.1))

            kdepo_vals = range(kdepo_low, kdepo_high, length = scan_points)
            d_vals = range(d_low, d_high, length = scan_points)
            theta_vals = range(theta_low, theta_high, length = scan_points)

            sse_kdepo_d = zeros(length(kdepo_vals), length(d_vals))
            for (i, k_val) in enumerate(kdepo_vals)
                for (j, d_val) in enumerate(d_vals)
                    sols_scan = simulate_depo_pulses(params; kdepo = k_val, thmax = thmax_opt, theta_mx = theta_mx_opt,
                        kdesorp = kdesorp_opt, blocking_fn = blocking_fn, D_override = d_val)
                    sse_kdepo_d[i, j] = total_lsq(datasets, sols_scan)
                end
            end
            write_scan_csv("Out_scan_kdepo_D.txt", kdepo_vals, d_vals, sse_kdepo_d; x_label = "kdepo", y_label = "D")

            sse_kdepo_theta = zeros(length(kdepo_vals), length(theta_vals))
            for (i, k_val) in enumerate(kdepo_vals)
                for (j, t_val) in enumerate(theta_vals)
                    sols_scan = simulate_depo_pulses(params; kdepo = k_val, thmax = thmax_opt, theta_mx = t_val,
                        kdesorp = kdesorp_opt, blocking_fn = blocking_fn)
                    sse_kdepo_theta[i, j] = total_lsq(datasets, sols_scan)
                end
            end
            write_scan_csv("Out_scan_kdepo_theta_mx.txt", kdepo_vals, theta_vals, sse_kdepo_theta; x_label = "kdepo", y_label = "theta_mx")

            if plot_scan
                heatmap(kdepo_vals, d_vals, sse_kdepo_d'; xlabel = "kdepo", ylabel = "D",
                    title = "SSE scan (kdepo vs D)")
                savefig("Out_scan_kdepo_D.png")
                heatmap(kdepo_vals, theta_vals, sse_kdepo_theta'; xlabel = "kdepo", ylabel = "theta_mx",
                    title = "SSE scan (kdepo vs theta_mx)")
                savefig("Out_scan_kdepo_theta_mx.png")
            end
        end

        return (; sols..., kdepo_opt, thmax_opt, theta_mx_opt, kdesorp_opt, params, k_fit, k_cft, eta, sse, aicbic)
    end

    "Convenience wrapper that uses the spherocylinder blocking function."
    function fit_ADEdepo_spherocyl2_pulse(; go_optim = false)
        return fit_ADEdepo_pulse(; go_optim, blocking_fn = B_RSA_nonSP_spherocyl2, particle_shape = :spherocyl)
    end
end
