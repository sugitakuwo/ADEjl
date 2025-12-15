module ADEsolute
    export ADE_solute_pulse, ADE_solute_step, intpsol

    using DomainSets
    using DataInterpolations
    using Interpolations
    using MethodOfLines
    using ModelingToolkit
    using OrdinaryDiffEq
    using Symbolics

    using ..PulseInputs: pulse_input_SP, pulse_input_step

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
