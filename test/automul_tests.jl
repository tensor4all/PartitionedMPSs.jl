using Test

using Random
using ITensors
using ITensorMPS

import PartitionedMPSs:
    PartitionedMPSs,
    PartitionedMPS,
    SubDomainMPS,
    makesitediagonal,
    extractdiagonal,
    project,
    elemmul,
    automul,
    default_cutoff,
    rearrange_siteinds

import FastMPOContractions as FMPOC

@testset "automul.jl" begin
    @testset "element-wise product" begin
        Random.seed!(1234)
        N = 5
        L = 10 # Bond dimension
        d = 2 # Local dimension 

        sites = [Index(d, "Qubit, n=$n") for n in 1:N]
        sites_vec = [[x] for x in sites]

        Ψ = ITensorMPS.convert(MPS, _random_mpo(sites_vec; linkdims=L))
        dummy_subdmps = SubDomainMPS(Ψ)

        proj_lev_l = 2 # Max projected index left tensor 
        proj_lev_r = 3 # Max projected index right tensor

        proj_l = vec([
            Dict(zip(sites, combo)) for
            combo in Iterators.product((1:d for _ in 1:proj_lev_l)...)
        ])

        proj_r = vec([
            Dict(zip(sites, combo)) for
            combo in Iterators.product((1:d for _ in 1:proj_lev_r)...)
        ])

        partΨ_l = PartitionedMPS(project.(Ref(Ψ), proj_l))
        partΨ_r = PartitionedMPS(project.(Ref(Ψ), proj_r))

        diag_dummy_l = makesitediagonal(dummy_subdmps, sites; baseplev=1)
        diag_dummy_r = makesitediagonal(dummy_subdmps, sites; baseplev=0)

        elemmul_dummy = extractdiagonal(
            PartitionedMPSs.contract(diag_dummy_l, diag_dummy_r; alg="zipup"), sites
        )

        element_prod = elemmul(partΨ_l, partΨ_r)
        mps_element_prod = MPS(element_prod)

        @test mps_element_prod ≈ MPS(elemmul_dummy)

        test_points = [[rand(1:d) for __ in 1:N] for _ in 1:1000]

        @test isapprox(
            [_evaluate(mps_element_prod, sites, p) for p in test_points],
            [_evaluate(Ψ, sites, p)^2 for p in test_points];
            atol=sqrt(default_cutoff()), # default_cutoff() = 1e-25 is the contraction cutoff
        )
    end

    @testset "matmul" begin
        N = 10
        d = 2
        L = 5

        sites_m = [Index(d, "Qubit, m=$m") for m in 1:N]
        sites_n = [Index(d, "Qubit, n=$n") for n in 1:N]
        sites_l = [Index(d, "Qubit, l=$l") for l in 1:N]
        sites_mn = collect(Iterators.flatten(collect.(zip(sites_m, sites_n))))
        sites_nl = collect(Iterators.flatten(collect.(zip(sites_n, sites_l))))
        final_sites = collect(Iterators.flatten(collect.(zip(sites_m, sites_l))))

        Ψ_l = ITensorMPS.convert(MPS, _random_mpo([[x] for x in sites_mn]; linkdims=L))
        Ψ_r = ITensorMPS.convert(MPS, _random_mpo([[x] for x in sites_nl]; linkdims=L))

        proj_lev_l = 4
        proj_lev_r = 6

        proj_l = vec([
            Dict(zip(sites_mn, combo)) for
            combo in Iterators.product((1:d for _ in 1:proj_lev_l)...)
        ])

        proj_r = vec([
            Dict(zip(sites_nl, combo)) for
            combo in Iterators.product((1:d for _ in 1:proj_lev_r)...)
        ])

        partΨ_l = PartitionedMPS(project.(Ref(Ψ_l), proj_l))
        partΨ_r = PartitionedMPS(project.(Ref(Ψ_r), proj_r))

        matmul = automul(
            partΨ_l, partΨ_r; alg="zipup", tag_row="m", tag_shared="n", tag_col="l"
        )
        mps_matmul = MPS(matmul)

        sites_mn_vec = collect(collect.(zip(sites_m, sites_n)))
        sites_nl_vec = collect(collect.(zip(sites_n, sites_l)))

        mpo_l = MPO(collect(rearrange_siteinds(Ψ_l, sites_mn_vec)))
        mpo_r = MPO(collect(rearrange_siteinds(Ψ_r, sites_nl_vec)))

        naive_matmul = FMPOC.contract_mpo_mpo(mpo_l, mpo_r; alg="naive")
        mps_naive_matmul = rearrange_siteinds(
            ITensorMPS.convert(MPS, naive_matmul), [[x] for x in final_sites]
        )

        @test mps_matmul ≈ mps_naive_matmul

        test_points = [[rand(1:d) for __ in 1:(2 * N)] for _ in 1:1000]

        @test isapprox(
            [_evaluate(mps_matmul, final_sites, p) for p in test_points],
            [_evaluate(mps_naive_matmul, final_sites, p) for p in test_points];
            atol=sqrt(default_cutoff()), # default_cutoff() = 1e-25 is the contraction cutoff
        )
    end
end
