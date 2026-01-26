module Blocking
    export B_RSA_SP, B_RSA_SP2, thmax_prolate_unoriented, thmax_prolate_average,
        B_RSA_nonSP_prolate, B_RSA_nonSP_spherocyl, B_RSA_nonSP_spherocyl2,
        B_RSA_nonSP_gamma2, B_none

    using Symbolics
    using SpecialFunctions

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

    "Blocking function using a provided shape factor gamma_p with transition at theta_t."
    function B_RSA_nonSP_gamma2(theta, thmax, gamma_p)
        theta_t = 0.34
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

    "No-blocking helper (ASF = 1)."
    B_none(theta, thmax) = 1.0

    @register_symbolic B_RSA_SP(theta, thmax)
    @register_symbolic B_RSA_SP2(theta, thmax)
    @register_symbolic B_RSA_nonSP_prolate(theta, thmax, As)
    @register_symbolic B_RSA_nonSP_spherocyl(theta, thmax, As)
    @register_symbolic B_RSA_nonSP_spherocyl2(theta, thmax, As)
    @register_symbolic B_RSA_nonSP_gamma2(theta, thmax, gamma_p)
    @register_symbolic B_none(theta, thmax)
end
