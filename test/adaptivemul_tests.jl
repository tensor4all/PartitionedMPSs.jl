import PartitionedMPSs:
    PartitionedMPSs,
    PartitionedMPS,
    SubDomainMPS,
    project,
    adaptivecontract,
    contract,
    Projector
import FastMPOContractions as FMPOC
import QuanticsGrids as QG
import TensorCrossInterpolation as TCI
import TCIAlgorithms as TCIA

using ITensors, ITensorMPS

Random.seed!(1234)

@testset "adaptivemul.jl" begin
    @testset "adaptivecontract" begin
        R = 8
        L = 5
        d = 2

        tol = 1e-5

        sites_x = [Index(d, "Qubit,x=$n") for n in 1:R]
        sites_y = [Index(d, "Qubit,y=$n") for n in 1:R]
        sites_s = [Index(d, "Qubit,s=$n") for n in 1:R]

        sites_l = collect(collect.(zip(sites_x, sites_s)))
        sites_r = collect(collect.(zip(sites_s, sites_y)))
        pordering = final_sites = collect(Iterators.flatten(zip(sites_x, sites_y)))

        mpo_l = _random_mpo(sites_l; linkdims=L)
        mpo_r = _random_mpo(sites_r; linkdims=L)

        proj_lev_l = 3
        proj_l = vec([
            Dict(zip(collect(Iterators.flatten(sites_l)), combo)) for
            combo in Iterators.product((1:d for _ in 1:proj_lev_l)...)
        ])

        proj_lev_r = 2
        proj_r = vec([
            Dict(zip(collect(Iterators.flatten(sites_r)), combo)) for
            combo in Iterators.product((1:d for _ in 1:proj_lev_r)...)
        ])

        partΨ_l = PartitionedMPS(project.(Ref(MPS(collect(mpo_l))), proj_l))
        partΨ_r = PartitionedMPS(project.(Ref(MPS(collect(mpo_r))), proj_r))

        part_normal = PartitionedMPSs.contract(partΨ_l, partΨ_r; cutoff=tol^2, alg="fit")
        part_adaptive = adaptivecontract(
            partΨ_l, partΨ_r, pordering; cutoff=tol^2, maxdim=23
        )

        @test MPS(part_adaptive) ≈ MPS(part_normal)

        naive_mpo = FMPOC.contract_mpo_mpo(mpo_l, mpo_r; cutoff=tol^2, alg="naive")

        @test MPS(part_adaptive) ≈ MPS(collect(naive_mpo))
    end

    @testset "2D Gaussians" begin

        # Integrand function
        gaussian(x, y) = exp(-1.0 * (x^2 + y^2))

        # Analytic solution
        analyticIntegral(x, y) = sqrt(π / 2) * exp(-1.0 * (x^2 + y^2))

        # Function parameters
        D = 2
        x_0 = 10.0

        # Simulation parameters
        R = 20
        unfoldingscheme = :fused
        mb = 25
        tol = 1e-9

        localdims = fill(2^D, R)
        sitedims = fill([2, 2], R)

        grid = QG.DiscretizedGrid{D}(
            R, Tuple(fill(-x_0, D)), Tuple(fill(x_0, D)); unfoldingscheme=unfoldingscheme
        )
        q_gauss = x -> gaussian(QG.quantics_to_origcoord(grid, x)...)
        patch_ordering = TCIA.PatchOrdering(collect(1:R))

        gauss_patch = reshape(
            TCIA.adaptiveinterpolate(
                TCIA.makeprojectable(Float64, q_gauss, localdims),
                patch_ordering;
                verbosity=0,
                maxbonddim=mb,
                tolerance=tol,
            ),
            sitedims,
        )

        sites_x = [Index(2, "Qubit,x=$n") for n in 1:R]
        sites_y = [Index(2, "Qubit,y=$n") for n in 1:R]
        sites_s = [Index(2, "Qubit,s=$n") for n in 1:R]

        sites_l = collect(collect.(zip(sites_x, sites_s)))
        sites_r = collect(collect.(zip(sites_s, sites_y)))
        pordering = final_sites = collect(Iterators.flatten(zip(sites_x, sites_y)))

        part_mps_l = PartitionedMPS(gauss_patch, sites_l)
        part_mps_r = PartitionedMPS(gauss_patch, sites_r)

        part_adaptive = adaptivecontract(
            part_mps_l, part_mps_r, pordering; cutoff=tol^2, maxdim=mb
        )

        adaptive_mps = PartitionedMPSs.rearrange_siteinds(
            MPS(part_adaptive), [[x] for x in final_sites]
        )

        N_err = 1000
        points = [(rand() * x_0 - x_0 / 2, rand() * x_0 - x_0 / 2) for _ in 1:N_err]
        quantics_fused_points = QG.origcoord_to_quantics.(Ref(grid), points)
        quantics_points = [
            QG.interleave_dimensions(QG.unfuse_dimensions(p, D)...) for
            p in quantics_fused_points
        ]

        adaptive_points = [
            (2x_0 / 2^R) * _evaluate(adaptive_mps, final_sites, p) for p in quantics_points
        ]
        analytic_points = [analyticIntegral(p...) for p in points]

        @test all(isapprox.(analytic_points, adaptive_points; atol=sqrt(tol)))
    end
end
