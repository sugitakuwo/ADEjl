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

"Utility helpers for pulse shape definitions."
module Utils
    export heaviside, hexp

    "Heaviside step function."
    function heaviside(t::Float64)::Float64
        return t < 0 ? 0 : (t == 0 ? 0.5 : 1)
    end

    "Gaussian-like step function."
    function hexp(x::Float64, dx::Float64)::Float64
        return x >= 0 ? exp(-dx^2 / x^2) : 0
    end
end

"Pulse input utilities for multiple smoothing strategies."
module PulseInputs
    export pulse_input_step, pulse_input_hexp, pulse_input_SP

    using ..Utils: heaviside, hexp
    using Symbolics

    "Pulse input function with a Heaviside step."
    function pulse_input_step(t, pulse_width)
        return heaviside(t) - heaviside(t - pulse_width)
    end

    "Pulse input function using the hexp smoothing."
    function pulse_input_hexp(t, pulse_width, dx)
        delt = abs(t - pulse_width / 2)
        h1 = hexp((pulse_width / 2 + dx / 2) - delt, dx)
        h2 = hexp(delt - (pulse_width / 2 - dx / 2), dx)
        return h1 / (h1 + h2)
    end

    "Pulse input based on the Smooth Profile method."
    function pulse_input_SP(t, pulse_width, dx)
        delt = t - pulse_width / 2.0
        return t < 0 ? 0 : (delt < 0 ? 1.0 : 0.5 * (1.0 + tanh((pulse_width / 2.0 - abs(delt)) / dx)))
    end

    @register_symbolic pulse_input_step(t, pulse_width)
    @register_symbolic pulse_input_hexp(t, pulse_width, dx)
    @register_symbolic pulse_input_SP(t, pulse_width, dx)
end

"Advection–dispersion solvers and fitting routines."
module ADEsolute
    export ADE_solute_pulse, ADE_solute_step, intpsol

    using DomainSets
    using DataInterpolations
    using Interpolations
    using MethodOfLines
    using ModelingToolkit
    using OrdinaryDiffEq

    using ..PulseInputs: pulse_input_SP, pulse_input_step
    using Symbolics

    "Interpolate a solution profile for later evaluation."
    function intpsol(x, table_x, table_y)
        interpolator = CubicSpline(table_y, table_x)
        return interpolator(x)
    end

    @register_symbolic intpsol(x, table_x::AbstractVector, table_y::AbstractVector)

    "Solve the ADE with a pulse boundary condition."
    function ADE_solute_pulse(; L = 5.3e-2, xmax = 1.0, tmax = 2.5, vp_in = 5.0e-4, D_in = 0.01,
        f_in = 1.0e-2, dx = 0.01, pulse_width = 0.2, delta = 0.02, solver = Tsit5())
        @parameters x t
        @parameters vp, Lc, Dsol
        @variables c(..)

        Dt = Differential(t)
        Dx = Differential(x)
        Dxx = Differential(x)^2

        eqs = [
            Dt(c(x, t)) ~ Dsol / (vp * Lc) * Dxx(c(x, t)) - Dx(c(x, t)),
        ]

        bcs = [
            c(x, 0.0) ~ 0.0,
            c(0.0, t) ~ pulse_input_SP(t, pulse_width, delta),
            Dx(c(xmax, t)) ~ 0.0,
        ]

        domains = [x ∈ Interval(0.0, xmax), t ∈ Interval(0.0, tmax)]

        @named pdesys = PDESystem(eqs, bcs, domains, [x, t], [c(x, t)], [vp => vp_in, Dsol => D_in, Lc => L])

        disc = MOLFiniteDifference([x => dx], t, approx_order = 6)
        prob = discretize(pdesys, disc)

        sol = solve(prob, solver)
        discrete_x = sol[x]
        discrete_t = sol[t]
        solc = sol[c(x, t)]

        return discrete_x, discrete_t, solc, sol
    end

    "Solve the ADE with a step boundary condition."
    function ADE_solute_step(; L = 5.3e-2, xmax = 1.0, tmax = 2.5, vp_in = 5.0e-4, D_in = 0.01,
        f_in = 1.0e-2, dx = 0.01, pulse_width = 0.2, delta = 0.02, solver = Tsit5())
        @parameters t x
        @parameters vp, Lc, f, Dsol
        @variables c(..)

        Dt = Differential(t)
        Dx = Differential(x)
        Dxx = Differential(x)^2

        eqs = [
            Dt(c(x, t)) ~ Dsol / (vp * Lc) * Dxx(c(x, t)) - Dx(c(x, t)),
        ]

        bcs = [
            c(x, 0.0) ~ 0.0,
            c(0.0, t) ~ 1.0,
            Dx(c(1, t)) ~ 0.0,
        ]

        domains = [x ∈ Interval(0.0, xmax), t ∈ Interval(0.0, tmax)]

        @named pdesys = PDESystem(eqs, bcs, domains, [x, t], [c(x, t)], [vp => vp_in, Dsol => D_in, Lc => L])

        disc = MOLFiniteDifference([x => dx], t, approx_order = 6)
        prob = discretize(pdesys, disc)

        sol = solve(prob, solver)
        discrete_x = sol[x]
        discrete_t = sol[t]
        solc = sol[c(x, t)]

        return discrete_x, discrete_t, solc, sol
    end
end

using .ADEsolute
export fit_ADEsolute_pulse

"Advection–dispersion with deposition and blocking models."
module ADEdeposition
    export B_RSA_SP, B_RSA_SP2, thmax_prolate_unoriented, thmax_prolate_average,
        B_RSA_nonSP_prolate, B_RSA_nonSP_spherocyl, B_RSA_nonSP_spherocyl2,
        ADE_depo_RSA_nonSP_pulse, Lsq_ADEdepo_pulse, fit_ADEdepo_pulse

    using DomainSets
    using DataInterpolations
    using Interpolations
    using MethodOfLines
    using ModelingToolkit
    using OrdinaryDiffEq
    using Symbolics

    using ..ADEsolute: intpsol
    using ..PulseInputs: pulse_input_SP

    "Blocking function based on the RSA model for spherical particles."
    function B_RSA_SP(theta, thmax)
        beta = 0.44 / thmax
        term1 = 1.0 - 4.0 * theta * beta
        term2 = 6.0 * sqrt(3.0) / pi * (theta * beta)^2
        term3 = (40.0 / (sqrt(3.0) * pi) - 176.0 / (3.0 * pi^2)) * (theta * beta)^3
        return term1 + term2 + term3
    end

    "Blocking function based on the RSA model for spherical particles (polynomial fit)."
    function B_RSA_SP2(theta, thmax)
        term1 = 1.0 + 0.812 * theta / thmax
        term2 = 0.426 * (theta / thmax)^2
        term3 = 0.0716 * (theta / thmax)^3
        return (term1 + term2 + term3) * (1 - theta / thmax)^3
    end

    "Maximum surface coverage for unoriented prolate particles."
    function thmax_prolate_unoriented(As)
        return 0.304 + 0.365 * As - 0.123 / As
    end

    "Average maximum surface coverage for prolate particles."
    function thmax_prolate_average(As)
        thmax_sideon = 0.622 * ((As + 1 / As - 1.997)^(0.0127)) * exp(-0.0274 * (As + 1 / As))
        return 0.5 * (0.547 / As + thmax_sideon)
    end

    "Blocking function for non-spherical prolate particles at low coverage."
    function B_RSA_nonSP_prolate(theta, thmax, As)
        C1 = 2.07 + 0.811 / As + 2.37 / As^2 - 1.25 / As^3
        C2 = (0.670 * As^2 - 0.301 * As + 3.88) / (0.283 * As^2 + As)
        term1 = 1.0 - C1 * theta / thmax
        term2 = C2 * (theta / thmax)^2
        return term1 + term2
    end

    "Blocking function for spherocylinders at low coverage."
    function B_RSA_nonSP_spherocyl(theta, thmax, As)
        gamma_p = (2 * As + pi - 2)^2 / (4 * pi * (As - 1 + pi / 4))
        C1 = 2 * (1 + gamma_p)
        C2_1 = (2 * As + pi - 2)^4 / (8 * pi^2 * (As - 1 + pi / 4)^2)
        C2_2 = (2 * As + pi - 2)^2 / (4 * pi * (As - 1 + pi / 4)) + 0.5
        term1 = 1.0 - C1 * theta / thmax
        term2 = (C2_1 + C2_2) * (theta / thmax)^2
        return term1 + term2
    end

    "Blocking function for spherocylinders with transition at theta_t."
    function B_RSA_nonSP_spherocyl2(theta, thmax, As)
        theta_t = 0.34
        gamma_p = (2 * As + pi - 2)^2 / (4 * pi * (As - 1 + pi / 4))
        if theta > theta_t
            term1 = (1 - theta_t)
            exp1 = -(1 + 2 * gamma_p) * theta_t / (1 - theta_t)
            exp2 = -gamma_p * (theta_t / (1 - theta_t))^2
            return term1 * exp(exp1 + exp2) * ((thmax - theta) / (thmax - theta_t))^4
        else
            term1 = (1 - theta)
            exp1 = -(1 + 2 * gamma_p) * theta / (1 - theta)
            exp2 = -gamma_p * (theta / (1 - theta))^2
            return term1 * exp(exp1 + exp2)
        end
    end

    @register_symbolic B_RSA_SP(theta, thmax)
    @register_symbolic B_RSA_SP2(theta, thmax)
    @register_symbolic B_RSA_nonSP_prolate(theta, thmax, As)
    @register_symbolic B_RSA_nonSP_spherocyl(theta, thmax, As)
    @register_symbolic B_RSA_nonSP_spherocyl2(theta, thmax, As)

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
        f_in = 1.0e-2, kdepo_in = 1.0e-2, Areap = 1.0e-5, C0_in = 6.58e19, theta_max = 1.0, Aspr = 10.0,
        dx = 0.01, pulse_width = 0.2, delta = 0.02, theta_init = nothing, x_init = nothing, solver = Tsit5())
        @parameters t x
        @parameters vp, Lc, f, C0, Dsol, kdepo, Ap, thmax, As
        @variables c(..), theta(..)

        Dt = Differential(t)
        Dx = Differential(x)
        Dxx = Differential(x)^2

        eqs = [
            Dt(c(x, t)) ~ Dsol / (vp * Lc) * Dxx(c(x, t)) - Dx(c(x, t)) - Lc / vp * f * kdepo * B_RSA_SP2(theta(x, t), thmax) * c(x, t),
            Dt(theta(x, t)) ~ Lc / vp * kdepo * B_RSA_SP2(theta(x, t), thmax) * Ap * C0 * c(x, t),
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
                As => Aspr, C0 => C0_in, thmax => theta_max])

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
        files = filter(f -> occursin(pattern, f), readdir())
        length(files) < 3 && error("Need at least three experimental files matching pattern=$(pattern)")
        data = [readdlm(files[i], ',', Float64, skipstart = 2) for i in 1:3]

        for arr in data
            arr[:, 2] ./= Absinput
            arr[:, 1] = (arr[:, 1] .- (tatmax - tp)) ./ tp
        end

        return [arr[0.0 .< arr[:, 1] .< tmax, :] for arr in data]
    end

    "Least squares objective for fitting deposition parameters."
    function Lsq_ADEdepo_pulse(p::Vector, exp_data, params)
        kdepo, thmax = p
        vp = params.vapp / params.fporo
        tp = params.L / vp
        D = params.alphaL * vp
        f = 3.0 * (1.0 - params.fporo) / (params.ac * params.fporo)
        pulse_width = params.tpulse / tp

        _, _, _, _, sol = ADE_depo_RSA_nonSP_pulse(L = params.L, xmax = params.xmax, tmax = params.tmax, vp_in = vp,
            D_in = D, f_in = f, kdepo_in = kdepo, Areap = params.Areap, C0_in = params.C0,
            theta_max = thmax, Aspr = params.Aspr, pulse_width = pulse_width, dx = params.dx, solver = params.solver)

        lsqsum = 0.0
        for ii in axes(exp_data, 1)
            solc_val = sol(exp_data[ii, 1], 1.0)
            lsqsum += (exp_data[ii, 2] - solc_val[1])^2
        end
        return lsqsum
    end

    "Fit deposition parameters and simulate three consecutive injections."
    function fit_ADEdepo_pulse(; go_optim = false)
        Absinput = 0.3468
        tatmax = 141.0
        tpulse = 20.0

        L = 0.053
        xmax = 1.0
        tmax = 2.5
        dc = 0.03e-2
        ac = 0.5 * dc
        vapp = 0.02e-2
        fporo = 0.383
        alphaL = 0.0007
        dx = 0.005
        solver = RadauIIA5()

        Lp = 87e-9
        ap = 0.5 * 7.3e-9
        bp = 0.5 * Lp
        Aspr = bp / ap
        Area_prolate = pi * bp * ap
        Area_Cyl = Lp * 2.0 * ap
        Area_ave = 0.5 * (Area_prolate + pi * ap^2)
        C0 = 2 * 9.15e19

        params = (; L, xmax, tmax, vapp, fporo, alphaL, ac, tpulse, dx, solver, Areap = Area_ave, C0, Aspr)

        vp = vapp / fporo
        tp = L / vp

        datasets = load_experiments(; Absinput, tatmax, tp)
        exp1, exp2, exp3 = datasets

        initial_x = [5.0e-7, 0.18]
        lower = [1.0e-7, 0.13]
        upper = [2.0e-6, 0.25]
        n_particles = 5
        options = Optim.Options(iterations = 10)

        if go_optim
            res = Optim.optimize(p -> Lsq_ADEdepo_pulse(p, exp1, params), initial_x, ParticleSwarm(lower, upper, n_particles), options)
            kdepo_opt, thmax_opt = Optim.minimizer(res)
        else
            kdepo_opt = 8.95e-7
            thmax_opt = 0.193
        end

        D = alphaL * vp
        f = 3.0 * (1.0 - fporo) / (ac * fporo)
        pulse_width = tpulse / tp

        xs1, ts1, solc1, soltheta1, sol1 = ADE_depo_RSA_nonSP_pulse(L = L, xmax = xmax, tmax = tmax, vp_in = vp, D_in = D,
            f_in = f, kdepo_in = kdepo_opt, Areap = Area_ave, C0_in = C0, theta_max = thmax_opt,
            pulse_width = pulse_width, dx = dx, solver = solver)

        xs2, ts2, solc2, soltheta2, sol2 = ADE_depo_RSA_nonSP_pulse(L = L, xmax = xmax, tmax = tmax, vp_in = vp, D_in = D,
            f_in = f, kdepo_in = kdepo_opt, Areap = Area_ave, C0_in = C0, theta_max = thmax_opt,
            pulse_width = pulse_width, dx = dx, theta_init = soltheta1, x_init = xs1, solver = solver)

        xs3, ts3, solc3, soltheta3, sol3 = ADE_depo_RSA_nonSP_pulse(L = L, xmax = xmax, tmax = tmax, vp_in = vp, D_in = D,
            f_in = f, kdepo_in = kdepo_opt, Areap = Area_ave, C0_in = C0, theta_max = thmax_opt,
            pulse_width = pulse_width, dx = dx, theta_init = soltheta2, x_init = xs2, solver = solver)

        outfile1 = "Out_exp1st_2nd_3rd_rescaled.txt"
        outfile2 = "Out_cal1st_injection.txt"
        outfile3 = "Out_cal2nd_injection.txt"
        outfile4 = "Out_cal3rd_injection.txt"

        open(outfile1, "w") do f_io
            println(f_io, "#Experimental tpv, c/c0 for 1st, 2nd, 3rd injection")
            for ii in axes(exp1, 1)
                println(f_io, exp1[ii, 1], " ", exp1[ii, 2], " ", exp2[ii, 1], " ", exp2[ii, 2], " ", exp3[ii, 1], " ", exp3[ii, 2])
            end
        end

        open(outfile2, "w") do f_io
            println(f_io, "#Calculated tpv, c/c0 for 1st injection")
            for ii in eachindex(ts1)
                println(f_io, ts1[ii], " ", solc1[end, ii])
            end
        end

        open(outfile3, "w") do f_io
            println(f_io, "#Calculated tpv, c/c0 for 2nd injection")
            for ii in eachindex(ts2)
                println(f_io, ts2[ii], " ", solc2[end, ii])
            end
        end

        open(outfile4, "w") do f_io
            println(f_io, "#Calculated tpv, c/c0 for 3rd injection")
            for ii in eachindex(ts3)
                println(f_io, ts3[ii], " ", solc3[end, ii])
            end
        end

        return (; xs1, ts1, solc1, soltheta1, xs2, ts2, solc2, soltheta2, xs3, ts3, solc3, soltheta3, kdepo_opt, thmax_opt, params)
    end
end

using .ADEdeposition
export fit_ADEdepo_pulse, ADE_depo_RSA_nonSP_pulse

"Least squares objective for dispersion length and porosity fitting."
function Lsq_ADEsolute_pulse(p::Vector, data, Lc, xmax, tmax, tatmax, vapp, dx_in, t_pulse, solverin)
    data = copy(data)
    alphaL = p[1]
    fporo = p[2]
    vp = vapp / fporo
    tp = Lc / vp
    D = alphaL * vp
    pulse_width_in = t_pulse / tp
    data[:, 1] = (data[:, 1] .- (tatmax - tp)) ./ tp
    data = data[0.0 .<= data[:, 1] .<= 2.5, :]

    xs, ts, solc, sol1 = ADE_solute_pulse(L = Lc, xmax = xmax, tmax = tmax, vp_in = vp, D_in = D, pulse_width = pulse_width_in, dx = dx_in, solver = solverin)

    lsqsum = 0.0
    for ii in axes(data, 1)
        solc_val = sol1(data[ii, 1], 1.0)
        lsqsum += (data[ii, 2] - solc_val[1])^2
    end
    return lsqsum
end

"Fit dispersion length and effective porosity using tracer experiment data."
function fit_ADEsolute_pulse(; go_optim = false)
    date = Dates.today()
    outfile1 = "Out_solute_exp_ave_rescaled_$(date).txt"
    outfile2 = "Out_calsolute_$(date).txt"
    outfile3 = "Out_optimized_disp_leng_fporo_$(date).txt"

    fn = readdir()
    fn = fn[map(x -> occursin(r"TL", x), fn)]
    filetracer1, filetracer2, filetracer3 = fn[1], fn[2], fn[3]
    Absinput = 0.6073
    exptracer1 = readdlm(filetracer1, ',', Float64, skipstart = 2)
    exptracer2 = readdlm(filetracer2, ',', Float64, skipstart = 2)
    exptracer3 = readdlm(filetracer3, ',', Float64, skipstart = 2)
    maxtime = minimum([maximum(exptracer1[:, 1]), maximum(exptracer2[:, 1]), maximum(exptracer3[:, 1])])

    exptracer1[:, 2] ./= Absinput
    exptracer2[:, 2] ./= Absinput
    exptracer3[:, 2] ./= Absinput
    exptracer1 = exptracer1[0.0 .<= exptracer1[:, 1] .<= maxtime, :]
    exptracer2 = exptracer2[0.0 .<= exptracer2[:, 1] .<= maxtime, :]
    exptracer3 = exptracer3[0.0 .<= exptracer3[:, 1] .<= maxtime, :]

    exptracer_mean = zeros(size(exptracer1))
    exptracer_std = zeros(size(exptracer1))

    for k in axes(exptracer_mean, 1)
        exptracer_mean[k, 1] = mean([exptracer1[k, 1]; exptracer2[k, 1]; exptracer3[k, 1]])
        exptracer_mean[k, 2] = mean([exptracer1[k, 2]; exptracer2[k, 2]; exptracer3[k, 2]])
        exptracer_std[k, 1] = std([exptracer1[k, 1]; exptracer2[k, 1]; exptracer3[k, 1]])
        exptracer_std[k, 2] = std([exptracer1[k, 2]; exptracer2[k, 2]; exptracer3[k, 2]])
    end

    mass_conserve, err = quadgk(x -> intpsol(x, exptracer_mean[:, 1], exptracer_mean[:, 2]), 0.0, maxtime)

    t_pulse = mass_conserve <= 20 ? 20 : (mass_conserve > 23 ? 23.0 : mass_conserve)
    tmax = 2.5
    xmax = 1.0
    Lc = 0.053
    vapp = 0.02e-2

    solverin = RadauIIA5()
    dx_in = 0.005

    maxindex = argmax(exptracer_mean[:, 2])
    tatmax = exptracer_mean[maxindex, 1]

    initial_x = [0.0005, 0.38]
    lower = [0.0001, 0.37]
    upper = [0.002, 0.43]
    n_particles = 7
    options = Optim.Options(iterations = 70)

    if go_optim
        res = Optim.optimize(p -> Lsq_ADEsolute_pulse(p, exptracer_mean, Lc, xmax, tmax, tatmax, vapp, dx_in, t_pulse, solverin), initial_x, ParticleSwarm(lower, upper, n_particles), options)
        alphaL_opt = Optim.minimizer(res)[1]
        fporo_opt = Optim.minimizer(res)[2]
    else
        alphaL_opt = 0.0006
        fporo_opt = 0.383
    end

    vp = vapp / fporo_opt
    tp = Lc / vp
    D = alphaL_opt * vp
    pulse_width_in = t_pulse / tp
    exptracer_mean[:, 1] = (exptracer_mean[:, 1] .- (tatmax - tp)) ./ tp
    exptracer_mean = exptracer_mean[0.0 .<= exptracer_mean[:, 1] .<= 2.5, :]

    xs, ts, solc, sol1 = ADE_solute_pulse(L = Lc, xmax = xmax, tmax = tmax, vp_in = vp, D_in = D, pulse_width = pulse_width_in, dx = dx_in, solver = solverin)

    slice = 42

    p1 = plot(xs, solc[:, slice], xlabel = "x", ylabel = "c/c0 (Cal.)", title = "t=$(ts[slice])")
    p2 = plot(ts, solc[end, :], xlabel = "t", ylabel = "c/c0", label = "Tracer (Cal.)")
    plot!(p2, exptracer_mean[:, 1], exptracer_mean[:, 2], xlabel = "t", ylabel = "c/c0", label = "Tracer (Exp.)")
    plot(p1, p2, size = (1500, 800))
    savefig("ADEsolute_pulse_alpha$(round(alphaL_opt, digits = 6))_fporo$(round(fporo_opt, digits = 6))_$(date).png")

    open(outfile1, "w") do f
        println(f, "#Experimental tpv, c/c0 (mean), stderr for tracer experiments")
        for ii in axes(exptracer_mean, 1)
            println(f, exptracer_mean[ii, 1], " ", exptracer_mean[ii, 2], " ", exptracer_std[ii, 2])
        end
    end
    open(outfile2, "w") do f
        println(f, "#Calculated tpv, c/c0 for tracer")
        for ii in eachindex(ts)
            println(f, ts[ii], " ", solc[end, ii])
        end
    end
    open(outfile3, "w") do f
        println(f, "#Optimized disp. leng., and eff. porosity based on tracer experiments")
        println(f, alphaL_opt)
        println(f, fporo_opt)
    end

    return (; xs, ts, solc, sol1, alphaL_opt, fporo_opt, mass_conserve, err)
end

end
