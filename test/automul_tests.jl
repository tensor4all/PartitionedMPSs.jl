using Test

using ITensors
import PartitionedMPSs:
    PartitionedMPSs, Projector, project, SubDomainMPS, projcontract, PartitionedMPS
import FastMPOContractions as FMPOC
using Quantics: Quantics

@testset "mul.jl" begin
    """
    Reconstruct 3D matrix
    """
    function _tomat3(a::MPS)
        sites = siteinds(a)
        N = length(sites)
        Nreduced = N ÷ 3
        sites_ = [sites[1:3:N]..., sites[2:3:N]..., sites[3:3:N]...]
        return reshape(Array(reduce(*, a), sites_), 2^Nreduced, 2^Nreduced, 2^Nreduced)
    end

    @testset "batchedmatmul" for T in [Float64]
        """
        C(x, z, k) = sum_y A(x, y, k) * B(y, z, k)
        """
        nbit = 2
        D = 2
        cutoff = 1e-25
        sx = [Index(2, "Qubit,x=$n") for n in 1:nbit]
        sy = [Index(2, "Qubit,y=$n") for n in 1:nbit]
        sz = [Index(2, "Qubit,z=$n") for n in 1:nbit]
        sk = [Index(2, "Qubit,k=$n") for n in 1:nbit]

        sites_a = collect(Iterators.flatten(zip(sx, sy, sk)))
        sites_b = collect(Iterators.flatten(zip(sy, sz, sk)))

        a = random_mps(T, sites_a; linkdims=D)
        b = random_mps(T, sites_b; linkdims=D)

        # Reference data
        a_arr = _tomat3(a)
        b_arr = _tomat3(b)
        ab_arr = zeros(T, 2^nbit, 2^nbit, 2^nbit)
        for k in 1:(2^nbit)
            ab_arr[:, :, k] .= a_arr[:, :, k] * b_arr[:, :, k]
        end

        a_ = PartitionedMPS([
            project(a, Projector(Dict(sx[1] => 1, sy[1] => 1))),
            project(a, Projector(Dict(sx[1] => 1, sy[1] => 2))),
            project(a, Projector(Dict(sx[1] => 2, sy[1] => 1))),
            project(a, Projector(Dict(sx[1] => 2, sy[1] => 2))),
        ])

        b_ = PartitionedMPS([
            project(b, Projector(Dict(sy[1] => 1, sz[1] => 1))),
            project(b, Projector(Dict(sy[1] => 1, sz[1] => 2))),
            project(b, Projector(Dict(sy[1] => 2, sz[1] => 1))),
            project(b, Projector(Dict(sy[1] => 2, sz[1] => 2))),
        ])

        @test a ≈ MPS(a_)
        @test b ≈ MPS(b_)

        ab = PartitionedMPSs.automul(
            a_, b_; tag_row="x", tag_shared="y", tag_col="z", alg="fit", cutoff
        )
        ab_ref = Quantics.automul(
            a, b; tag_row="x", tag_shared="y", tag_col="z", alg="fit", cutoff
        )

        @test MPS(ab) ≈ ab_ref rtol = 10 * sqrt(cutoff)
    end
end
