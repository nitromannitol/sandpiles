# sandpiles

**Deprecated.** This repository is superseded by
[arbitrarydim_sandpiles](https://github.com/nitromannitol/arbitrarydim_sandpiles), which
contains up-to-date sandpile code for CPUs and for CUDA GPUs (through CUDA.jl), in any
dimension.

The code here is an OpenCL implementation for Julia 0.5 and 0.6 that stabilizes
two-dimensional Abelian sandpiles quickly on a GPU. The function

```julia
output_sand, output_odometer = GPU_stabilize(input_sand, domain)
```

in [`sandpile_init.jl`](sandpile_init.jl) takes an `N × M` array of integers `input_sand` and
an `N × M` array `domain`, where `domain[i,j] = -1` marks a site outside the domain. It
returns the stable sandpile and the odometer, the number of times each site topples during
the stabilization, both as `N × M` arrays. [`example.jl`](example.jl) shows three uses: a
single source, the all-4s sandpile on a square, and the all-4s sandpile on a diamond domain.

## Running

The code was written for Julia 0.5 and 0.6 with the OpenCL.jl package of that time and needs
an OpenCL device. It is not maintained and does not run on current Julia: it uses syntax
removed in Julia 1.0 (for example `Vector{Float32}(1)` in `sandpile_init.jl`), and OpenCL.jl
did not appear to work with newer Julia versions. Use
[arbitrarydim_sandpiles](https://github.com/nitromannitol/arbitrarydim_sandpiles) instead.

## Authors and citation

The code is by **Ahmed Bou-Rabee**. The up-to-date code in
[arbitrarydim_sandpiles](https://github.com/nitromannitol/arbitrarydim_sandpiles) accompanies
*Dynamic dimensional reduction in the Abelian sandpile*; see its README for the citation.
