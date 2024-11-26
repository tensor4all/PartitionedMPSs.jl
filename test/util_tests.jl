using Test

using ITensors
import PartitionedMPSs:
    PartitionedMPSs,
    Projector,
    project,
    SubDomainMPS,
    projcontract,
    PartitionedMPS,
    rearrange_siteinds,
    makesitediagonal,
    extractdiagonal
import FastMPOContractions as FMPOC

@testset "util.jl" begin
    @testset "rearrange_siteinds" begin
        N = 3
        sitesx = [Index(2, "x=$n") for n in 1:N]
        sitesy = [Index(2, "y=$n") for n in 1:N]
        sitesz = [Index(2, "z=$n") for n in 1:N]
        sites = collect(collect.(zip(sitesx, sitesy, sitesz)))

        Ψ = MPS(collect(_random_mpo(sites)))

        prjΨ = SubDomainMPS(Ψ)
        prjΨ1 = project(prjΨ, Dict(sitesx[1] => 1))

        sitesxy = collect(collect.(zip(sitesx, sitesy)))
        sites_rearranged = Vector{Index{Int}}[]
        for i in 1:N
            push!(sites_rearranged, sitesxy[i])
            push!(sites_rearranged, [sitesz[i]])
        end
        prjΨ1_rearranged = rearrange_siteinds(prjΨ1, sites_rearranged)

        @test reduce(*, MPS(prjΨ1)) ≈ reduce(*, MPS(prjΨ1_rearranged))
        @test PartitionedMPSs.siteinds(prjΨ1_rearranged) == sites_rearranged
    end
end
