module ADEdeposition
    export ADE_depo_RSA_nonSP_pulse, Lsq_ADEdepo_pulse, fit_ADEdepo_pulse

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
    using QuadGK

    using ..ADEsolute: intpsol
    using ..PulseInputs: pulse_input_SP
    using ..Blocking: B_RSA_SP2, B_RSA_nonSP_spherocyl2

    "Dispatch-safe evaluation for blocking functions with or without aspect ratio."
    blocking_value(blocking_fn, theta, thmax, As) = applicable(blocking_fn, theta, thmax, As) ?
        blocking_fn(theta, thmax, As) : blocking_fn(theta, thmax)

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
        dx = 0.01, pulse_width = 0.2, delta = 0.02, theta_init = nothing, x_init = nothing, solver = Tsit5(),
        blocking_fn = B_RSA_SP2)
        @parameters t x
        @parameters vp, Lc, f, C0, Dsol, kdepo, Ap, thmax, As
        @variables c(..), theta(..)

        Dt = Differential(t)
        Dx = Differential(x)
        Dxx = Differential(x)^2

        eqs = [
            Dt(c(x, t)) ~ Dsol / (vp * Lc) * Dxx(c(x, t)) - Dx(c(x, t)) -
            Lc / vp * f * kdepo * blocking_value(blocking_fn, theta(x, t), thmax, As) * c(x, t),
            Dt(theta(x, t)) ~ Lc / vp * kdepo * blocking_value(blocking_fn, theta(x, t), thmax, As) * Ap * C0 * c(x, t),
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
    function Lsq_ADEdepo_pulse(p::Vector, exp_data, params; blocking_fn = B_RSA_SP2)
        kdepo, thmax = p
        vp = params.vapp / params.fporo
        tp = params.L / vp
        D = params.alphaL * vp
        f = 3.0 * (1.0 - params.fporo) / (params.ac * params.fporo)
        pulse_width = params.tpulse / tp

        _, _, _, _, sol = ADE_depo_RSA_nonSP_pulse(L = params.L, xmax = params.xmax, tmax = params.tmax, vp_in = vp,
            D_in = D, f_in = f, kdepo_in = kdepo, Areap = params.Areap, C0_in = params.C0,
            theta_max = thmax, Aspr = params.Aspr, pulse_width = pulse_width, dx = params.dx,
            solver = params.solver, blocking_fn = blocking_fn)

        lsqsum = 0.0
        for ii in axes(exp_data, 1)
            solc_val = sol(exp_data[ii, 1], 1.0)
            lsqsum += (exp_data[ii, 2] - solc_val[1])^2
        end
        return lsqsum
    end

    "Fit deposition parameters and simulate three consecutive injections."
    function fit_ADEdepo_pulse(; go_optim = false, blocking_fn = B_RSA_SP2)
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
            res = Optim.optimize(p -> Lsq_ADEdepo_pulse(p, exp1, params; blocking_fn), initial_x,
                ParticleSwarm(lower, upper, n_particles), options)
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
            pulse_width = pulse_width, dx = dx, solver = solver, blocking_fn = blocking_fn)

        xs2, ts2, solc2, soltheta2, sol2 = ADE_depo_RSA_nonSP_pulse(L = L, xmax = xmax, tmax = tmax, vp_in = vp, D_in = D,
            f_in = f, kdepo_in = kdepo_opt, Areap = Area_ave, C0_in = C0, theta_max = thmax_opt,
            pulse_width = pulse_width, dx = dx, theta_init = soltheta1, x_init = xs1, solver = solver,
            blocking_fn = blocking_fn)

        xs3, ts3, solc3, soltheta3, sol3 = ADE_depo_RSA_nonSP_pulse(L = L, xmax = xmax, tmax = tmax, vp_in = vp, D_in = D,
            f_in = f, kdepo_in = kdepo_opt, Areap = Area_ave, C0_in = C0, theta_max = thmax_opt,
            pulse_width = pulse_width, dx = dx, theta_init = soltheta2, x_init = xs2, solver = solver,
            blocking_fn = blocking_fn)

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

    "Convenience wrapper that uses the spherocylinder blocking function."
    function fit_ADEdepo_spherocyl2_pulse(; go_optim = false)
        return fit_ADEdepo_pulse(; go_optim, blocking_fn = B_RSA_nonSP_spherocyl2)
    end
end
