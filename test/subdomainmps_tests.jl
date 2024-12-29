using Test

using ITensors

using Random

import PartitionedMPSs:
    PartitionedMPSs,
    Projector,
    project,
    SubDomainMPS,
    rearrange_siteinds,
    makesitediagonal,
    extractdiagonal

@testset "subdomainmps.jl" begin
    @testset "SubDomainMPS" begin
        Random.seed!(1)
        N = 3
        sitesx = [Index(2, "x=$n") for n in 1:N]
        sitesy = [Index(2, "y=$n") for n in 1:N]
        sites = collect(collect.(zip(sitesx, sitesy)))
        Ψ = MPS(collect(_random_mpo(sites)))
        prjΨ = SubDomainMPS(Ψ)

        prjΨ1 = project(prjΨ, Dict(sitesx[1] => 1))
        prjΨ2 = project(prjΨ, Dict(sitesx[1] => 2))

        Ψreconst = MPS(prjΨ1) + MPS(prjΨ2)

        @test ITensors.norm(prjΨ1) ≈ ITensors.norm(MPS(prjΨ1))

        @test Ψreconst ≈ Ψ
    end

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

    @testset "makesitediagonal and extractdiagonal" begin
        N = 3
        sitesx = [Index(2, "x=$n") for n in 1:N]
        sitesy = [Index(2, "y=$n") for n in 1:N]
        sitesz = [Index(2, "z=$n") for n in 1:N]

        sitesxy_vec = [[x, y] for (x, y) in zip(sitesx, sitesy)]
        sitesz_vec = [[z] for z in sitesz]
        sites = [x for pair in zip(sitesxy_vec, sitesz_vec) for x in pair]

        Ψ = MPS(collect(_random_mpo(sites)))

        prjΨ = SubDomainMPS(Ψ)
        prjΨ1 = project(prjΨ, Dict(sitesx[1] => 1))

        prjΨ1_diagonalz = makesitediagonal(prjΨ1, "y")
        sites_diagonalz = Iterators.flatten(siteinds(prjΨ1_diagonalz))

        psi_diag = prod(prjΨ1_diagonalz.data)
        psi = prod(prjΨ1.data)

        @test extractdiagonal(prjΨ1_diagonalz, "y") ≈ prjΨ1

        diag_ok = true
        offdiag_ok = true

        for indval in eachindval(sites_diagonalz...)
            ind = first.(indval)
            val = last.(indval)

            index_dict = Dict{Index{Int},Vector{Int}}()
            for (i, el) in enumerate(ind)
                baseind = noprime(el)
                if haskey(index_dict, baseind)
                    push!(index_dict[baseind], i)
                else
                    index_dict[baseind] = [i]
                end
            end
            repeated_indices = [is for is in values(index_dict) if length(is) > 1]

            isdiagonalelement = all(allequal(val[i] for i in is) for is in repeated_indices)

            if isdiagonalelement
                nondiaginds = unique(noprime(i) => v for (i, v) in indval)
                diag_ok = diag_ok && (psi_diag[indval...] == psi[nondiaginds...])
            else
                offdiag_ok = offdiag_ok && iszero(psi_diag[indval...])
            end
        end

        @test diag_ok
        @test offdiag_ok
    end
end
