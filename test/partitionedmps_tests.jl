using Test

using ITensors
using ITensorMPS
using Random

import PartitionedMPSs: PartitionedMPSs, Projector, project, SubDomainMPS, PartitionedMPS

@testset "partitionedmps.jl" begin
    @testset "two blocks" begin
        Random.seed!(1234)
        N = 3
        sitesx = [Index(2, "x=$n") for n in 1:N]
        sitesy = [Index(2, "y=$n") for n in 1:N]

        sites = collect(collect.(zip(sitesx, sitesy)))

        Ψ = MPS(collect(_random_mpo(sites)))

        prjΨ = SubDomainMPS(Ψ)

        prjΨ1 = project(prjΨ, Dict(sitesx[1] => 1))
        prjΨ2 = project(prjΨ, Dict(sitesx[1] => 2))

        @test_throws ErrorException PartitionedMPS([prjΨ, prjΨ1])
        @test_throws ErrorException PartitionedMPS([prjΨ1, prjΨ1])

        # Iterator and length
        @test length(PartitionedMPS(prjΨ1)) == 1
        @test length([(k, v) for (k, v) in PartitionedMPS(prjΨ1)]) == 1

        Ψreconst = PartitionedMPS(prjΨ1) + PartitionedMPS(prjΨ2)
        @test Ψreconst[Projector(sitesx[1] => 1)] ≈ prjΨ1
        @test Ψreconst[Projector(sitesx[1] => 2)] ≈ prjΨ2
        @test MPS(Ψreconst) ≈ Ψ
        @test ITensors.norm(Ψreconst) ≈ ITensors.norm(MPS(Ψreconst))

        # Summation
        coeffs = (1.1, 0.9)
        @test MPS(+(PartitionedMPS(prjΨ1), PartitionedMPS(prjΨ2); coeffs=coeffs)) ≈
            coeffs[1] * MPS(prjΨ1) + coeffs[2] * MPS(prjΨ2)
    end

    @testset "two blocks (general key)" begin
        Random.seed!(1234)
        N = 3
        sitesx = [Index(2, "x=$n") for n in 1:N]
        sitesy = [Index(2, "y=$n") for n in 1:N]

        sites = collect(collect.(zip(sitesx, sitesy)))

        Ψ = MPS(collect(_random_mpo(sites)))

        prjΨ = SubDomainMPS(Ψ)

        prjΨ1 = project(prjΨ, Dict(sitesx[1] => 1))
        prjΨ2 = project(prjΨ, Dict(sitesx[1] => 2))

        a = PartitionedMPS(prjΨ1)
        b = PartitionedMPS(prjΨ2)

        @test MPS(2 * a) ≈ 2 * MPS(a) rtol = 1e-13
        @test MPS(a * 2) ≈ 2 * MPS(a) rtol = 1e-13
        @test MPS((a + b) + 2 * (b + a)) ≈ 3 * Ψ rtol = 1e-13
        @test MPS((a + b) + 2 * (b + a)) ≈ 3 * Ψ rtol = 1e-13
    end

    @testset "add" begin
        Random.seed!(1234)
        N = 3
        d = 10
        sites = [Index(d, "x=$n") for n in 1:N]
        Ψ = MPS(collect(_random_mpo([[s] for s in sites])))

        projectors = [Projector(sites[1] => d_) for d_ in 1:d]
        coeffs = rand(length(projectors))
        prjΨs = [project(Ψ, p) for p in projectors]

        @test MPS(+([PartitionedMPS(x) for x in prjΨs]...; coeffs=coeffs)) ≈ +([c * MPS(x) for (c, x) in zip(coeffs, prjΨs)]...; alg="directsum")
    end

    @testset "truncate" begin
        for seed in [1, 2, 3, 4, 5]
            Random.seed!(seed)
            N = 10
            D = 10 # Bond dimension
            d = 10 # local dimension
            cutoff_global = 1e-4

            sites = [[Index(d, "n=$n")] for n in 1:N]

            Ψ = 100 * MPS(collect(_random_mpo(sites; linkdims=D)))

            partmps = PartitionedMPS([project(Ψ, Dict(sites[1][1] => d_)) for d_ in 1:d])
            partmps_truncated = PartitionedMPSs.truncate(partmps; cutoff=cutoff_global)

            diff =
                ITensorMPS.dist(MPS(partmps_truncated), MPS(partmps))^2 /
                ITensorMPS.norm(MPS(partmps))^2
            @test diff < cutoff_global
        end
    end
end
