#!/usr/bin/env julia
using Pkg
project_root = joinpath(@__DIR__, "..")
Pkg.activate(project_root)
Pkg.instantiate()

module ADE_MOL
using LinearAlgebra, SpecialFunctions, Statistics
using Plots, RecipesBase, Printf, DelimitedFiles
using OrdinaryDiffEq
using Optimization, OptimizationOptimJL
using ForwardDiff
using Dates

#Plotting function, macro to plot concentration profile 
@recipe function plot(range::StepRangeLen, sol::ODESolution)
    ts = sol.t
    N = length(ts)
    xs = collect(range[2:end-1])
    M = length(xs)

#    layout := @layout [c]#[c c]#

    t_out = collect(0:0.2:2)
    k=1
    for i in 1:N
        if ts[i] ≥ t_out[k]
           @series begin
#                subplot := 1
                ylabel --> "c"
                xlabel --> "x"
                ylims --> (0, 1)
                label --> @sprintf "t = %1.1f" ts[i]
                xs, sol.u[i][1:M]
            end
            k += 1
        end
    end
    xlims --> (0, 1)
end

#Utility Functions to generate pulse input  模拟不同脉冲输入
module Utils
    export heaviside, hexp
#---heaviside Step function
    function heaviside(t::Float64)::Float64
        return t < 0 ? 0 : (t == 0 ? 0.5 : 1)
    end
#---Gassian-like step function
    function hexp(x::Float64, dx::Float64)::Float64
        return x >= 0 ? exp(-dx^2 / x^2) : 0
    end
end# module Utils

#Pulse input function with step function
module PulseInputs
    export pulse_input_step, pulse_input_hexp, pulse_input_SP
    using ..Utils: heaviside, hexp
    using Symbolics, SpecialFunctions

    #Pulse input function with (heaviside) step function
    function pulse_input_step(t,pulse_width)
        return heaviside(t)-heaviside(t-pulse_width)
    end

    #Pulse input function with hexp
    function pulse_input_hexp(t,pulse_width,dx)
        delt = abs(t - pulse_width/2)
        h1 = hexp((pulse_width/2+dx/2)-delt,dx)
        h2 = hexp(delt-(pulse_width/2-dx/2),dx)
        return h1/(h1+h2)
    end

    #Pulse input function with Smooth Profile method
    function pulse_input_SP(t,pulse_width,dx)
        delt = abs(t - pulse_width/2)
        func1 = tanh((pulse_width/2-delt)/dx)+1.0
        return 0.5*func1
    end
    @register_symbolic pulse_input_step(t,pulse_width)
    @register_symbolic pulse_input_hexp(t,pulse_width,dx)
    @register_symbolic pulse_input_SP(t,pulse_width,dx)
end# module PulseInputs
#----------------------------------------------

module Blocking  #定义阻塞函数B(θ) SP就为球型，nonSP为非球型，一共五种B(θ)
    export B_RSA_SP, B_RSA_SP2, B_RSA_nonSP_spherocyl_lowtheta, B_RSA_nonSP_spherocyl2, B_RSA_nonSP_prolate_lowtheta
    export thmax_prolate_unoriented, thmax_prolate_average
    using Symbolics, SpecialFunctions
    
#---基于RSA的球型SP阻塞函数
    function B_RSA_SP(theta,thmax)
        beta=0.44/thmax
        term1 = 1.0-4.0*theta*beta
        term2 = 6.0*sqrt(3.0)/pi*(theta*beta)^2
        term3 = (40.0/(sqrt(3.0)*pi)-176.0/(3.0*pi^2))*(theta*beta)^3
        return term1+term2+term3
    end
    function B_RSA_SP2(theta,thmax)
        term1 = 1.0+0.812*theta/thmax
        term2 = 0.426*(theta/thmax)^2
        term3 = 0.0716*(theta/thmax)^3
        return (term1+term2+term3)*(1-theta/thmax)^3
    end

#---最大覆盖率thmax的两种计算方式
    function thmax_prolate_unoriented(As)
        return 0.304+0.365*As-0.123/As
    end

    function thmax_prolate_average(As)
        thmax_sideon=0.622*((As+1/As-1.997)^(0.0127))*exp(-0.0274*(As+1/As))
        return 0.5*(0.547/As+thmax_sideon)
    end
#---Assuming low coverage limit, theta < 0.6 when As: aspect ratio ~ 10
#---基于RSA的非球型nonSP阻塞函数
    function B_RSA_nonSP_prolate_lowtheta(theta,thmax,As)
        C1=2.07+0.811/As+2.37/As^2-1.25/As^3
        C2=(0.670*As^2-0.301*As+3.88)/(0.283*As^2+As)
        term1 = 1.0-C1*theta/thmax
        term2 = C2*(theta/thmax)^2
        return term1+term2
    end

#---Assuming low coverage limit, theta < 0.6 when As,Aspr: aspect ratio ~ 10
    function B_RSA_nonSP_spherocyl_lowtheta(theta,thmax,As)
        gamma_p=(2*As+pi-2)^2/(4*pi*(As-1+pi/4))
        C1=2*(1+gamma_p)
        C2_1=(2*As+pi-2)^4/(8*pi^2*(As-1+pi/4)^2)
        C2_2=(2*As+pi-2)^2/(4*pi*(As-1+pi/4))+1/2
        term1 = 1.0-C1*theta/thmax
        term2 = (C2_1+C2_2)*(theta/thmax)^2
        return term1+term2
    end

    function B_RSA_nonSP_spherocyl2(theta,thmax,As)
        theta_t=0.34
        theta_val = theta#Symbolics.value(theta)
        thmax_val = thmax#Symbolics.value(thmax)
        gamma_p=(2*As+pi-2)^2/(4*pi*(As-1+pi/4))
        if theta_val > theta_t
            term1=(1-theta_t)
            exp1=-(1+2*gamma_p)*theta_t/(1-theta_t)
            exp2=-gamma_p*(theta_t/(1-theta_t))^2
            return term1*exp(exp1+exp2)*((thmax_val-theta_val)/(thmax_val-theta_t))^4
        else
            term1=(1-theta_val)
            exp1=-(1+2*gamma_p)*theta_val/(1-theta_val)
            exp2=-gamma_p*(theta_val/(1-theta_val))^2
            return term1*exp(exp1+exp2)
        end
    end
    @register_symbolic B_RSA_SP(theta,thmax)
    @register_symbolic B_RSA_SP2(theta,thmax)
    @register_symbolic B_RSA_nonSP_prolate_lowtheta(theta,thmax,As)
    @register_symbolic B_RSA_nonSP_spherocyl_lowtheta(theta,thmax,As)
    @register_symbolic B_RSA_nonSP_spherocyl2(theta,thmax,As)
end# module Blocking

#ADE运算
module ADEmain
    export ADE_depo_RSA_SP2_pulse, ADE_depo_RSA_spherocyl2_pulse #设置两种 ADE+RSA模型函数
    #MOL package has been registered!
    using ModelingToolkit, MethodOfLines, OrdinaryDiffEq, DomainSets
    using Interpolations, DataInterpolations
    using Symbolics, SpecialFunctions

    using ..PulseInputs: pulse_input_SP, pulse_input_hexp, pulse_input_step
    using ..Blocking: B_RSA_SP, B_RSA_SP2, B_RSA_nonSP_prolate_lowtheta, B_RSA_nonSP_spherocyl_lowtheta, B_RSA_nonSP_spherocyl2
    
#---Interpolation function for solutions and theta
#(intpsol::DataInterpolations.AbstractInterpolation)(x::Symbolics.Num) = Symbolics.SymbolicUtils.term(intpsol,x)
    function intpsol(x,table_x,table_y)
        interpolator=CubicSpline(table_y,table_x)
        output=interpolator(x)
    end
    @register_symbolic intpsol(x,table_x::AbstractVector,table_y::AbstractVector)

    function theta_intp(x,t,sol_in,dx)
        A_x1 = 0.0:dx:1.0
#    f(x1) = sol_in(t,x1)[2]
        A = sol_in(t,A_x1)[2] # [f(x1) for x1 in A_x1]
        itp = interpolate(A, BSpline(Cubic(Line(OnGrid()))))
        sitp1 = scale(itp, A_x1)
        return sitp1(x)
    end
    @register_symbolic theta_intp(x,t,sol_in,dx)

#定义球型和非球星ADE
#ADE with pulse input/deposition with the rate of kdepo and RSA Blocking function (spheres)
#Uapp: approach velocity, D_in: diffusion/dispersion coefficient, dx: spatial step size
    function ADE_depo_RSA_SP2_pulse(;L=5.3e-2,xmax=1.0,tmax=2.5,vp_in=5.0e-4,D_in=0.01,f_in=1.0e-2,kdepo_in=1.0e-2,Areap=1.0e-5,C0_in=6.58e19,theta_max=1.0,Aspr=10.0,dx=0.01,pulse_width=0.2,delta=0.02,theta_in=0.0,x_in=0.0,solver=Tsit5())
        @parameters t x
        @parameters vp, Lc, f, C0, Dsol, kdepo, Ap, thmax, As
        @variables c(..), theta(..)

        Dt=Differential(t)
        Dx=Differential(x)
        Dxx=Differential(x)^2

        eqs=[
            Dt(c(x,t))~Dsol/(vp*Lc)*Dxx(c(x,t))-Dx(c(x,t))-Lc/vp*f*kdepo*B_RSA_SP2(theta(x,t),thmax)*c(x,t),
            Dt(theta(x,t))~Lc/vp*kdepo*B_RSA_SP2(theta(x,t),thmax)*Ap*C0*c(x,t),
        ]
        if typeof(theta_in) != typeof(0.0)
    #       println(theta_in[:,end])
#    error()
            bcs1=[
                c(x,0.0)~0.0,
                c(0.0,t)~pulse_input_SP(t,pulse_width,delta),
                Dx(c(1,t))~0.0,
                theta(x,0)~intpsol(x,x_in,theta_in[:,end])
            ]
        else
            bcs1=[
                c(x,0.0)~0.0,
                c(0.0,t)~pulse_input_SP(t,pulse_width,delta),
                Dx(c(1,t))~0.0,
                theta(x,0)~0.0
            ]
        end

        domains1=[x ∈ Interval(0.0, xmax), t ∈ Interval(0.0, tmax)]

        @named pdesys=PDESystem(eqs,
                        bcs1,
                        domains1,
                        [x,t],
                        [c(x,t),theta(x,t)],
                        [vp=>vp_in,
                        Dsol=>D_in,
                        Lc=>L,
                        f=>f_in,
                        kdepo=>kdepo_in,
                        Ap=>Areap,
                        As=>Aspr,
                        C0=>C0_in,
                        thmax=>theta_max])

        disc=MOLFiniteDifference([x=>dx],t,approx_order=6)
        prob=discretize(pdesys,disc)

        sol1=solve(prob,solver)
        discrete_x=sol1[x]
        discrete_t=sol1[t]
        solc=sol1[c(x,t)]
        soltheta=sol1[theta(x,t)]

        return discrete_x, discrete_t, solc, soltheta, sol1
    end
#定义非球型的ADE公式
    #ADE with pulse input/deposition with the rate of kdepo and RSA Blocking function (spheres)
#Uapp: approach velocity, D_in: diffusion/dispersion coefficient, dx: spatial step size, Aspr: aspect ratio
    function ADE_depo_RSA_spherocyl2_pulse(;L=5.3e-2,xmax=1.0,tmax=2.5,vp_in=5.0e-4,D_in=0.01,f_in=1.0e-2,kdepo_in=1.0e-2,Areap=1.0e-5,C0_in=6.58e19,theta_max=1.0,Aspr=10.0,dx=0.01,pulse_width=0.2,delta=0.02,theta_in=0.0,x_in=0.0,solver=Tsit5())
        @parameters t x
        @parameters vp, Lc, f, C0, Dsol, kdepo, Ap, thmax, As
        @variables c(..), theta(..)

        Dt=Differential(t)
        Dx=Differential(x)
        Dxx=Differential(x)^2

        eqs=[
            Dt(c(x,t))~Dsol/(vp*Lc)*Dxx(c(x,t))-Dx(c(x,t))-(Lc/vp)*f*kdepo*B_RSA_nonSP_spherocyl2(theta(x,t),thmax,Aspr)*c(x,t),
            Dt(theta(x,t))~(Lc/vp)*kdepo*B_RSA_nonSP_spherocyl2(theta(x,t),thmax,Aspr)*Ap*C0*c(x,t),
        ]

        if typeof(theta_in) != typeof(0.0)
    #       println(theta_in[:,end])
#    error()
            bcs1=[
                c(x,0.0)~0.0,
                c(0.0,t)~pulse_input_SP(t,pulse_width,delta),
                Dx(c(1,t))~0.0,
                theta(x,0)~intpsol(x,x_in,theta_in[:,end])
            ]
        else
            bcs1=[
                c(x,0.0)~0.0,
                c(0.0,t)~pulse_input_SP(t,pulse_width,delta),
                Dx(c(1,t))~0.0,
                theta(x,0)~0.0
            ]
        end

        domains1=[x ∈ Interval(0.0, xmax), t ∈ Interval(0.0, tmax)]

        @named pdesys=PDESystem(eqs,
                        bcs1,
                        domains1,
                        [x,t],
                        [c(x,t),theta(x,t)],
                        [vp=>vp_in,
                        Dsol=>D_in,
                        Lc=>L,
                        f=>f_in,
                        kdepo=>kdepo_in,
                        Ap=>Areap,
                        As=>Aspr,
                        C0=>C0_in,
                        thmax=>theta_max])

        disc=MOLFiniteDifference([x=>dx],t,approx_order=6)
        prob=discretize(pdesys,disc)

        sol1=solve(prob,solver)
        discrete_x=sol1[x]
        discrete_t=sol1[t]
        solc=sol1[c(x,t)]
        soltheta=sol1[theta(x,t)]

        return discrete_x, discrete_t, solc, soltheta, sol1
    end
end# module ADEmain

using .ADEmain
using .Blocking
export fit_ADEdepo_spherocyl2_pulse

#运用ADE方程解和阻塞函数来拟合实验数据
function Lsq_ADEdepo_pulse(p::Vector, data,Lc,xmax,tmax,vp,D,f,Area_Cyl,C0,As,dx_in,pulse_width_in,solverin;theta=0.0,xsol=0.0)
#p[1]: kdepo, p[2]: thmax
#Define parameters to be optimized
kdepo=p[1]
thmax=p[2]
#Solve ADE to optimize parameters
# 更改为调用 spherocyl2 函数，确保使用非球形逻辑
xs, ts, solc, soltheta, sol1 = ADE_depo_RSA_spherocyl2_pulse(L=Lc,xmax=xmax,tmax=tmax,vp_in=vp,D_in=D,f_in=f,kdepo_in=kdepo,Areap=Area_Cyl,C0_in=C0,theta_max=thmax,Aspr=As,pulse_width=pulse_width_in,dx=dx_in,theta_in=theta,x_in=xsol,solver=solverin)
lsqsum=0.0
for ii in 1:length(data[:,1])
    solc=sol1(data[ii,1],1.0)
    lsqsum = lsqsum + (data[ii,2]-solc[1])^2
end
return lsqsum
end

function fit_ADEdepo_spherocyl2_pulse()
date=Dates.today()
outfile1="Out_CNC_exp1st_2nd_3rd_rescaled.txt"
outfile2="Out_cal1st_injection_SP2_$(date).txt"
outfile3="Out_cal2nd_injection_SP2_$(date).txt"
outfile4="Out_cal3rd_injection_SP2_$(date).txt"
#-- Open and read the results for tracer experiments ----
fn=readdir()
fn=fn[map(x->occursin(r"CN",x),fn)]
#println(fn)
file1=fn[1]
file2=fn[2]
file3=fn[3]
Absinput=0.1612#Abs for input cnc
exp1=readdlm(file1, ',', Float64, skipstart=2)
exp2=readdlm(file2, ',', Float64, skipstart=2)
exp3=readdlm(file3, ',', Float64, skipstart=2)

### CNC Particle parameters ###
Lp=87e-9 # m, TEM length
ap=0.5*7.3e-9 # m, TEM radius
bp=0.5*Lp # m, TEM length*0.5
Aspr=bp/ap # Aspect ratio = L/(2*ap), As
println("Aspect ratio =", Aspr)
thmax1=thmax_prolate_unoriented(Aspr)
thmax2=thmax_prolate_average(Aspr)
println("thmax1 =", thmax1)
println("thmax2 =", thmax2)
#error()
rho_s=1500 #? kg/m^3 density ?
Cyl_Vol=pi*ap*ap*Lp # m^3
Area_Cyl=Lp*2.0*ap#Particle area,forcylinder [m2]
Area_ave=0.5*(pi*bp*ap+pi*ap^2)#Average particle area, for prolate spheroids [m2]
C0=7.323e19#6.58e19# Particle initial number concentration [1/m3]
println("Area_Cyl =", Area_Cyl)
println("Area_ave =", Area_ave)
println("Area_prolate =", pi*bp*ap)
#error()

#Column parameters
pH=4.10
dc=0.03e-2#Collector diameter [m]
ac=0.5*dc#Collector radius [m]
tpulse=20.0#pulse width [s]
tmax=2.5#Scaled maximum time = Maximum Pore Volume [-]
xmax=1.0#Scaled column length [-]

#Choose a solver to solve the PDE system
solverin=RadauIIA5()#TRBDF2()#Rodas4()
dx_in=0.005#Spatial step size [m]

#Should be determined from experimental setup
# !!!alphaL_opt, fporo should be determined from tracer experiments!!
alphaL_opt=0.000926#Dispersion length (Dispersivity) [m],
Lc=0.053#Column length [m], should we use optimized ones from tracer experiments?
fporo=0.38 #Porosity [-]
f=3.0*(1.0-fporo)/(ac*fporo)#bare matrix surface area per pore volume [1/m]
vapp=0.01899e-2#Average approach velocity [m/s], should be determined from measured flow rate Q!!
vp=vapp/fporo##Average interstitial velocity [m/s]
tp=Lc/vp#time for 1 pore volume
D=alphaL_opt*vp#(Should be)Optimixed Dispersion coefficient [m2/s]
pulse_width=tpulse/tp#Pulse width in scaled time [-]
println("alphaL, fporo, vp, tp =",alphaL_opt," ",fporo," ",vp," ",tp)

#Scale and extract data to fit
exp1[:,2]=exp1[:,2]/Absinput
exp2[:,2]=exp2[:,2]/Absinput
exp3[:,2]=exp3[:,2]/Absinput

maxindex1st=argmax(exp1[:,2])
maxindex2nd=argmax(exp2[:,2])
maxindex3rd=argmax(exp3[:,2])
tatmax1st=exp1[maxindex1st,1]#149.0#time at Cmax
tatmax2nd=exp2[maxindex2nd,1]
tatmax3rd=exp3[maxindex3rd,1]
tatmax=tatmax1st#mean([tatmax1st,tatmax2nd,tatmax3rd])
exp1[:,1]=(exp1[:,1].-(tatmax-tp))/tp
exp2[:,1]=(exp2[:,1].-(tatmax-tp))/tp
exp3[:,1]=(exp3[:,1].-(tatmax-tp))/tp
exp1=exp1[0.0.<=exp1[:,1].<=2.5,:]
exp2=exp2[0.0.<=exp2[:,1].<=2.5,:]
exp3=exp3[0.0.<=exp3[:,1].<=2.5,:]
#exp_ave=1/3*(exp1+exp2+exp3)
println("Maximum Abs/Abs_in ~ C/C0 for each pulse =",maximum(exp1[:,2]), "at time $(tatmax1st) ", maximum(exp2[:,2])," at time $(tatmax2nd)  ",maximum(exp3[:,2])," at time $(tatmax3rd)")
println("Averaged maximum =",1/3*(maximum(exp1[:,2])+maximum(exp2[:,2])+maximum(exp3[:,2])))#Why?

#Initial guess and boundaries to optimize
initial_x=[5.0e-7,0.18]#(kdepo, thmax) at initial guess
lower = [2.0e-9,0.001]#Should change lower/upper bounds for each parameter?
upper = [2.0e-6,0.4]#
p = [1,1]
n_particles = 5#3, 5, 7
options = Optim.Options(iterations=50)#20, 50, 70
go_optim = true #true / false
kdepo_test = 5.0e-8
thmax_test = 0.45

if go_optim
#Optimize the parameters for each pulse
#1st pulse
#obj1 = (x,p) -> Lsq_ADEdepo_pulse(x, exp1,Lc,xmax,tmax,vp,D,f,Area_Cyl,C0,Aspr,dx_in,pulse_width,solverin)
#optf1 = OptimizationFunction(obj1)
#prob1 = OptimizationProblem(optf1, initial_x, lb=lower, ub=upper)
#sol1 = solve(prob1, ParticleSwarm())
    res1 = Optim.optimize(p -> Lsq_ADEdepo_pulse(p,exp1,Lc,xmax,tmax,vp,D,f,Area_Cyl,C0,Aspr,dx_in,pulse_width,solverin), initial_x, ParticleSwarm(lower,upper,n_particles),options)
    println("Optimized (kdepo, thmax) for 1st pulse =",Optim.minimizer(res1)[1]," ",Optim.minimizer(res1)[2]," and initial guess = ", initial_x)
#update initial guess
    initial_x[1]=Optim.minimizer(res1)[1]
    initial_x[2]=Optim.minimizer(res1)[2]
#error()
#2nd pulse
#obj2 = (x,p) -> Lsq_ADEdepo_pulse(x, exp2,Lc,xmax,tmax,vp,D,f,Area_Cyl,C0,Aspr,dx_in,pulse_width,solverin)
#optf2 = OptimizationFunction(obj2)
#prob2 = OptimizationProblem(optf2, initial_x, p, lb=lower, ub=upper)
#sol2 = solve(prob2, ParticleSwarm())
    xs, ts, solc, soltheta, sol1 = ADE_depo_RSA_spherocyl2_pulse(L=Lc,xmax=xmax,tmax=tmax,vp_in=vp,D_in=D,f_in=f,kdepo_in=initial_x[1],Areap=Area_Cyl,C0_in=C0,theta_max=initial_x[2],pulse_width=pulse_width,dx=dx_in,solver=solverin)
    res2 = Optim.optimize(p -> Lsq_ADEdepo_pulse(p,exp2,Lc,xmax,tmax,vp,D,f,Area_Cyl,C0,Aspr,dx_in,pulse_width,solverin,theta=soltheta,xsol=xs), initial_x, ParticleSwarm(lower,upper,n_particles),options)
    println("Optimized (kdepo, thmax) for 2nd pulse =",Optim.minimizer(res2)[1]," ",Optim.minimizer(res2)[2]," and initial guess = ", initial_x)
#3rd pulse
#obj3 = (x,p) -> Lsq_ADEdepo_pulse(x, exp3,Lc,xmax,tmax,vp,D,f,Area_Cyl,C0,Aspr,dx_in,pulse_width,solverin)
#optf3 = OptimizationFunction(obj3)
#prob3 = OptimizationProblem(optf3, initial_x, p, lb=lower, ub=upper)
#sol3 = solve(prob3, ParticleSwarm())
    xs2, ts2, solc2, soltheta2, sol2 = ADE_depo_RSA_spherocyl2_pulse(L=Lc,xmax=xmax,tmax=tmax,vp_in=vp,D_in=D,f_in=f,kdepo_in=initial_x[1],Areap=Area_Cyl,C0_in=C0,theta_max=initial_x[2],pulse_width=pulse_width,dx=dx_in,theta_in=soltheta,x_in=xs,solver=solverin)
    res3 = Optim.optimize(p -> Lsq_ADEdepo_pulse(p,exp3,Lc,xmax,tmax,vp,D,f,Area_Cyl,C0,Aspr,dx_in,pulse_width,solverin,theta=soltheta,xsol=xs), initial_x, ParticleSwarm(lower,upper,n_particles),options)
    println("Optimized (kdepo, thmax) for 3rd pulse =",Optim.minimizer(res3)[1]," ",Optim.minimizer(res3)[2]," and initial guess = ", initial_x)

    thmax=mean([Optim.minimizer(res1)[2],Optim.minimizer(res2)[2],Optim.minimizer(res3)[2]])# Maximum coverage [-]
    std_thmax=std([Optim.minimizer(res1)[2],Optim.minimizer(res2)[2],Optim.minimizer(res3)[2]])

    kdepo=mean([Optim.minimizer(res1)[1],Optim.minimizer(res2)[1],Optim.minimizer(res3)[1]])# Deposition rate coefficient [m/s]
    std_kdepo=std([Optim.minimizer(res1)[1],Optim.minimizer(res2)[1],Optim.minimizer(res3)[1]])
    println("Averaged kdepo, thmax =",kdepo,"+-", std_kdepo,"[m/s]" ,thmax,"+-", std_thmax,"[-]")
else
    kdepo = kdepo_test
    thmax = thmax_test
end

xs, ts, solc, soltheta, sol1 = ADE_depo_RSA_spherocyl2_pulse(L=Lc,xmax=xmax,tmax=tmax,vp_in=vp,D_in=D,f_in=f,kdepo_in=kdepo,Areap=Area_Cyl,C0_in=C0,theta_max=thmax,pulse_width=tpulse/tp,dx=dx_in,solver=solverin)
xs2, ts2, solc2, soltheta2, sol2 = ADE_depo_RSA_spherocyl2_pulse(L=Lc,xmax=xmax,tmax=tmax,vp_in=vp,D_in=D,f_in=f,kdepo_in=kdepo,Areap=Area_Cyl,C0_in=C0,theta_max=thmax,pulse_width=tpulse/tp,dx=dx_in,theta_in=soltheta,x_in=xs,solver=solverin)
xs3, ts3, solc3, soltheta3, sol3 = ADE_depo_RSA_spherocyl2_pulse(L=Lc,xmax=xmax,tmax=tmax,vp_in=vp,D_in=D,f_in=f,kdepo_in=kdepo,Areap=Area_Cyl,C0_in=C0,theta_max=thmax,pulse_width=tpulse/tp,dx=dx_in,theta_in=soltheta2,x_in=xs2,solver=solverin)

slice = 30
println("Value of f =", f)
println("Value of t_pulse/tp =", tpulse/tp)
println("size of solc", size(solc))
println(solc[end,:])

p1 = plot(xs,solc[:,slice],xlabel="x",ylabel="c/c0")
#p2 = plot(ts,solc[end,:],xlabel="t",ylabel="c/c0")
#p2 = plot(ts,sol1(ts,1.0)[1],xlabel="t",ylabel="c/c0")
#plot!(p2, ts2,sol2(ts2,1.0)[1],xlabel="t",ylabel="c/c0")
#plot!(p2, ts3,sol3(ts3,1.0)[1],xlabel="t",ylabel="c/c0")
p2 = plot(ts,solc[end,:],xlabel="t",ylabel="c/c0")
plot!(p2, ts2,solc2[end,:],xlabel="t",ylabel="c/c0")
plot!(p2, ts3,solc3[end,:],xlabel="t",ylabel="c/c0")
plot!(p2, exp1[:,1],exp1[:,2],xlabel="t",ylabel="c/c0", label="1st injection (Exp.)")
plot!(p2, exp2[:,1],exp2[:,2],xlabel="t",ylabel="c/c0", label="2nd injection (Exp.)")
plot!(p2, exp3[:,1],exp3[:,2],xlabel="t",ylabel="c/c0", label="3rd injection (Exp.)")
p3 = plot(xs,soltheta[:,end],xlabel="x",ylabel="theta")
plot!(p3, xs2 ,soltheta2[:,end],xlabel="x",ylabel="theta")
plot!(p3, xs3 ,soltheta3[:,end],xlabel="x",ylabel="theta")
plot(p1,p2,p3,size=(1500,800))
savefig("ADEdepo_pulse_kdepo$(round(kdepo*1e-7,digits=6))e-7_thmax$(round(thmax,digits=6))_$(date).png")

#open(outfile1,"w") do f
    #println(f, "#Experimental tpv, c/c0 for 1st, 2nd, 3rd injection")
    #for ii in 1:length(exp1[:,1])
        #println(f, exp1[ii,1]," ",exp1[ii,2]," ",exp2[ii,1]," ",exp2[ii,2]," ",exp3[ii,1], " ",exp3[ii,2])
    #end
#end
#open(outfile2,"w") do f
    #println(f, "#Calculated tpv, c/c0 for 1st injection")
    #for ii in 1:length(ts[:])
        #println(f, ts[ii]," ", solc[end,ii])
    #end
#end
#open(outfile3,"w") do f
    #println(f, "#Calculated tpv, c/c0 for 2nd injection")
    #for ii in 1:length(ts2[:])
        #println(f, ts2[ii]," ", solc2[end,ii])
    #end
#end
#open(outfile4,"w") do f
    #println(f, "#Calculated tpv, c/c0 for 3rd injection")
    #for ii in 1:length(ts3[:])
        #println(f, ts3[ii]," ", solc3[end,ii])
    #end
#end

#error()
end

end#module ADE_MOL
using .ADE_MOL

if abspath(PROGRAM_FILE) == @__FILE__
    fit_ADEdepo_spherocyl2_pulse()
end
