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
