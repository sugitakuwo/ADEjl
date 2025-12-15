using Dates
using DelimitedFiles
using Optim
using Plots
using QuadGK
using Statistics

using .ADEsolute
using .ADEdeposition: ADE_depo_RSA_nonSP_pulse

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

    xs, ts, solc, sol1 = ADE_solute_pulse(L = Lc, xmax = xmax, tmax = tmax, vp_in = vp, D_in = D, pulse_width = pulse_width_in,
        dx = dx_in, solver = solverin)

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
        res = Optim.optimize(p -> Lsq_ADEsolute_pulse(p, exptracer_mean, Lc, xmax, tmax, tatmax, vapp, dx_in, t_pulse, solverin),
            initial_x, ParticleSwarm(lower, upper, n_particles), options)
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

    xs, ts, solc, sol1 = ADE_solute_pulse(L = Lc, xmax = xmax, tmax = tmax, vp_in = vp, D_in = D, pulse_width = pulse_width_in,
        dx = dx_in, solver = solverin)

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
