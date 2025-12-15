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
