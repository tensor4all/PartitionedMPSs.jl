function ITensors.contract(M1::ProjMPS, M2::ProjMPS; kwargs...)::Union{ProjMPS, Nothing}
    if !hasoverlap(M1.projector, M2.projector)
        return nothing
    end
    proj = _projector_after_contract(M1, M2)
    Ψ = FMPOC.contract_mpo_mpo(MPO(collect(M1.data)), MPO(collect(M2.data)); kwargs...)
    return project(ProjMPS(Ψ), proj)
end


# Figure out `projector` after contracting ProjMPS objects
function _projector_after_contract(M1::ProjMPS, M2::ProjMPS)
    sites1 = Set(Iterators.flatten(siteinds(M1)))
    sites2 = Set(Iterators.flatten(siteinds(M2)))

    external_sites = setdiff(union(sites1, sites2), intersect(sites1, sites2))

    proj = deepcopy(M1.projector.data)

    for s in external_sites
        if isprojectedat(M1, s)
            proj[s] = M1.projector[s]
        end
        if isprojectedat(M2, s)
            proj[s] = M2.projector[s]
        end
    end

    return Projector(proj)
end


"""
Project two ProjMPS objects to `proj` before contracting them.
"""
function projcontract(M1::ProjMPS, M2::ProjMPS, proj::Projector; kwargs...)::Union{Nothing,ProjMPS}
    # Project M1 and M2 to `proj` before contracting
    M1 = project(M1, proj)
    M2 = project(M2, proj)
    return ITensors.contract(M1, M2; kwargs...)
end


"""
Project two ProjMPS objects to `proj` before contracting them.
The results are summed.
"""
function projcontract(M1::AbstractVector{ProjMPS}, M2::AbstractVector{ProjMPS}, proj::Projector; kwargs...)
    results = Vector{ProjMPS}[]
    for M1_ in M2, M2_ in M2
        r = projcontract(M1_, M2_, proj; kwargs...)
        if r !== nothing
            push!(results, r)
        end
    end

    if isempty(results)
        return nothing
    end

    if length(results) == 1
        return results[1]
    end

    # Add all results.
    # DO NOT USE DENSITYMATRIX ALGORITHM BECAUSE IT SUFFERS FROM NUMERICAL ERRORS
    # In the future, we may implement a more stable algorithm based on density matrix + fitting
    results_sum = deepcopy(results[1])
    for r in results[2:end]
        results_sum = _add(results_sum, r; kwargs...)
    end

    return results_sum
end

function _add(a::ProjMPS, b::ProjMPS; cutoff=0.0, maxdim=typemax(Int))::ProjMPS
    ab_MPS = +(a.data, b.data; alg="directsum")
    truncate!(ab_MPS; maxdim, cutoff)
    return project(ProjMPS(ab_MPS), a.projector | b.projector)
end