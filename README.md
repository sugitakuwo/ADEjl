# ADEjl

Reusable Julia utilities to solve advection–dispersion equations with MethodOfLines.jl and related tooling.

## Layout
- `src/ADE_MOL.jl`: Core module containing pulse definitions, PDE solvers, deposition/blocking models, and fitting helpers.
- `scripts/run_fit.jl`: Entrypoint for tracer (solute) fitting. Pass `--optimize` to enable parameter optimization.
- `scripts/run_depo.jl`: Entrypoint for deposition fitting with optional `--optimize`.

## Getting started
1. Ensure Julia 1.9+ is available.
2. Run the example script:
   ```bash
   julia scripts/run_fit.jl
   # or optimize parameters
   julia scripts/run_fit.jl --optimize
   ```

The script will read tracer experiment CSV files containing `TL` in their filenames, fit or simulate the ADE response, plot calculated versus experimental profiles, and write summary outputs.

For deposition fitting, supply three experimental CSV files that include `CN` in their filenames and run:

```bash
julia scripts/run_depo.jl
# or optimize deposition parameters
julia scripts/run_depo.jl --optimize
```
