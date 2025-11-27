using Test
import PartitionedMPSs:
    PartitionedMPSs,
    Projector,
    project,
    SubDomainMPS,
    adaptive_patching,
    PartitionedMPS,
    maxbonddim
import FastMPOContractions as FMPOC
using ITensors
using ITensorMPS
using Random
import TensorCrossInterpolation as TCI
import QuanticsGrids as QG

include("_util.jl")

# first example from the patching paper
function examplefunc(r)
    r1 = (0.2, 0.2)
    r2 = (0.8, 0.8)
    A = 10^4
    σ = 1e-1
    k = 10^3
    return A * exp(-norm(r .- r1)^2 / (2σ^2)) * sin(k * norm(r .- r1)) +
           A / 2 * exp(-norm(r .- r2)^2 / (σ^2 / 2)) * sin(k * norm(r .- r2) / 2)
end

function tci_examplefunc(R)
    grid = QG.DiscretizedGrid{2}(R; unfoldingscheme=:interleaved)
    qf(x) = examplefunc(QG.quantics_to_origcoord(grid, x))
    localdims = fill(2, 2 * R)

    Npivots = 5
    Random.seed!(1)
    firstpivots = [[rand(1:dim) for dim in localdims] for _ in 1:Npivots]
    firstpivots = map(p -> TCI.optfirstpivot(qf, localdims, p), firstpivots)
    tci_func, _ = TCI.crossinterpolate2(Float64, qf, localdims, firstpivots; tolerance=1e-7)

    sitesx = [Index(2, "Qubit,x=$n") for n in 1:R]
    sitesy = [Index(2, "Qubit,y=$n") for n in 1:R]

    sites_ = collect.(zip(sitesx, sitesy))
    sites = collect(Iterators.flatten(sites_))

    mps = MPS(TCI.TensorTrain(tci_func); sites)
    sdmps = SubDomainMPS(mps)
    pmps = PartitionedMPS([sdmps])

    return sites, pmps
end

@testset "patching.jl" begin
    @testset "adaptive_patching" begin
        Random.seed!(1234)

        R = 3
        sitesx = [Index(2, "Qubit,x=$n") for n in 1:R]
        sitesy = [Index(2, "Qubit,y=$n") for n in 1:R]

        sites = collect(collect.(zip(sitesx, sitesy)))

        subdmps = SubDomainMPS(_random_mpo(sites; linkdims=20))

        sites_ = collect(Iterators.flatten(sites))
        partmps = PartitionedMPS(
            adaptive_patching(subdmps, sites_; maxdim=10, cutoff=1e-25)
        )

        @test length(values((partmps))) > 1

        @test MPS(partmps) ≈ MPS(subdmps) rtol = 1e-12
    end

    @testset "adaptive_patching PartitionedMPS" begin
        R = 15
        sites, pmps = tci_examplefunc(R)
        pmps_partitioned = adaptive_patching(pmps, sites; maxdim=100, cutoff=1e-10)

        @test maximum(maxbonddim, values(pmps_partitioned)) <= 100
        @test length(values(pmps_partitioned)) > 1
        @test MPS(pmps_partitioned) ≈ MPS(pmps) rtol = 6e-5
    end
end
