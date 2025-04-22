using Test

using ITensors

import TensorCrossInterpolation as TCI
import TCIAlgorithms as TCIA
using TCIITensorConversion
import PartitionedMPSs: PartitionedMPSs

conversion_file = normpath(joinpath(dirname(pathof(PartitionedMPSs)), "bak/conversion.jl"))
include(conversion_file)

@testset "conversion.jl" begin
    @testset "TCIA.ProjTensorTrain => SubDomainMPS" begin
        N = 4
        χ = 2
        bonddims = [1, χ, χ, χ, 1]
        @assert length(bonddims) == N + 1

        localdims1 = [2, 2, 2, 2]
        localdims2 = [3, 3, 3, 3]
        sitedims = [[x, y] for (x, y) in zip(localdims1, localdims2)]
        localdims = collect(prod.(sitedims))

        tt = TCI.TensorTrain([
            rand(bonddims[n], localdims1[n], localdims2[n], bonddims[n + 1]) for n in 1:N
        ])

        for (n, tensor) in enumerate(tt)
            size(tensor)[2:(end - 1)] == sitedims[n]
        end

        # Projection
        prj = TCIA.Projector([[1, 1], [0, 0], [0, 0], [0, 0]], sitedims)
        prjtt = TCIA.project(TCIA.ProjTensorTrain{Float64}(tt), prj)

        sitesx = [Index(localdims1[n], "x=$n") for n in 1:N]
        sitesy = [Index(localdims2[n], "y=$n") for n in 1:N]
        sites = collect(collect.(zip(sitesx, sitesy)))

        prjmps = SubDomainMPS(prjtt, sites)

        @test MPS(prjmps) ≈ MPS(prjtt.data; sites=sites)
    end

    @testset "TCIA.ProjTTContainer => PartitionedMPS" begin
        N = 4
        χ = 2
        bonddims = [1, χ, χ, χ, 1]
        @assert length(bonddims) == N + 1

        localdims1 = [2, 2, 2, 2]
        localdims2 = [3, 3, 3, 3]
        sitedims = [[x, y] for (x, y) in zip(localdims1, localdims2)]
        localdims = collect(prod.(sitedims))

        tt = TCI.TensorTrain([
            rand(bonddims[n], localdims1[n], localdims2[n], bonddims[n + 1]) for n in 1:N
        ])

        for (n, tensor) in enumerate(tt)
            size(tensor)[2:(end - 1)] == sitedims[n]
        end

        # Projection
        prj1 = TCIA.Projector([[1, 1], [0, 0], [0, 0], [0, 0]], sitedims)
        prjtt1 = TCIA.project(TCIA.ProjTensorTrain{Float64}(tt), prj1)

        prj2 = TCIA.Projector([[1, 2], [0, 0], [0, 0], [0, 0]], sitedims)
        prjtt2 = TCIA.project(TCIA.ProjTensorTrain{Float64}(tt), prj2)

        prjttcontainer = TCIA.ProjTTContainer([prjtt1, prjtt2])

        sitesx = [Index(localdims1[n], "x=$n") for n in 1:N]
        sitesy = [Index(localdims2[n], "y=$n") for n in 1:N]
        sites = collect(collect.(zip(sitesx, sitesy)))

        bmps = PartitionedMPS(prjttcontainer, sites)

        @test MPS(bmps) ≈ +(
            MPS(prjtt1.data; sites=sites), MPS(prjtt2.data; sites=sites); alg="directsum"
        )
    end
end
