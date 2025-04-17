using Distributed
using BenchmarkTools
using Random

nworkers = 2

if nworkers > nprocs() - 1
    addprocs(nworkers)
end

library_dir = normpath(joinpath(dirname(pathof(PartitionedMPSs))))

@everywhere begin
    using Pkg
    Pkg.activate(library_dir)
    Pkg.instantiate()
    Pkg.precompile()
end

@everywhere begin
    import PartitionedMPSs:
        PartitionedMPSs,
        contract,
        PartitionedMPS,
        SubDomainMPS,
        Projector,
        project,
        _add,
        projcontract
end

using ITensors, ITensorMPS

random_mpo_file = normpath(joinpath(dirname(pathof(PartitionedMPSs)), "../test/_util.jl"))

Random.seed!(1234)
R = 10
d = 2
L = 5

sites_x = [Index(d, "Qubit,x=$x") for x in 1:R]
sites_y = [Index(d, "Qubit,y=$y") for y in 1:R]
sites_s = [Index(d, "Qubit,s=$s") for s in 1:R]

sites_xs = collect(collect.(zip(sites_x, sites_s)))
sites_sy = collect(collect.(zip(sites_s, sites_y)))

sites_xs_flat = collect(Iterators.flatten(sites_xs))
sites_sy_flat = collect(Iterators.flatten(sites_sy))

Ψ_l = ITensorMPS.convert(MPS, _random_mpo(sites_xs; linkdims=L))
Ψ_r = ITensorMPS.convert(MPS, _random_mpo(sites_sy; linkdims=L))

proj_lev_l = 4
proj_l = vec([
    Dict(zip(sites_xs_flat, combo)) for
    combo in Iterators.product((1:d for _ in 1:proj_lev_l)...)
])

proj_lev_r = 6
proj_r = vec([
    Dict(zip(sites_sy_flat, combo)) for
    combo in Iterators.product((1:d for _ in 1:proj_lev_r)...)
])

partΨ_l = PartitionedMPS(project.(Ref(Ψ_l), proj_l))
partΨ_r = PartitionedMPS(project.(Ref(Ψ_r), proj_r))

# ------------------------------------------------------------------
# Warm‑up (compile both code paths) --------------------------------
# ------------------------------------------------------------------
println("warming up …")
serial_warm = contract(partΨ_l, partΨ_r; parallel=:serial)
dist_warm = contract(partΨ_l, partΨ_r; parallel=:distributed)
@assert MPS(serial_warm) ≈ MPS(dist_warm) # sanity check

# ------------------------------------------------------------------
# Benchmark --------------------------------------------------------
# ------------------------------------------------------------------
println("\nbenchmarking …")
serial_time = @belapsed contract($partΨ_l, $partΨ_r; parallel=:serial)
dist_time = @belapsed contract($partΨ_l, $partΨ_r; parallel=:distributed)

println("\n---------------- results ----------------")
println("workers          : $(nprocs() - 1)")
println("serial time      : $(round(serial_time;  digits = 4)) s")
println("distributed time : $(round(dist_time;    digits = 4)) s")
println("speed‑up         : $(round(serial_time / dist_time; digits = 2))×")
println("-----------------------------------------")
