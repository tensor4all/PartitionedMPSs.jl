# just for backward compatibility...
_alg_map = Dict(
    ITensors.Algorithm(alg) => alg for alg in ["directsum", "densitymatrix", "fit", "naive"]
)

function contract(
    M1::SubDomainMPS, M2::SubDomainMPS; alg, kwargs...
)::Union{SubDomainMPS,Nothing}
    if !hasoverlap(M1.projector, M2.projector)
        return nothing
    end
    proj, _ = _projector_after_contract(M1, M2)

    alg_str::String = alg isa String ? alg : _alg_map[alg]
    Ψ = FMPOC.contract_mpo_mpo(
        MPO(collect(M1.data)), MPO(collect(M2.data)); alg=alg_str, kwargs...
    )
    return project(SubDomainMPS(Ψ), proj)
end

# Figure out `projector` after contracting SubDomainMPS objects
function _projector_after_contract(M1::SubDomainMPS, M2::SubDomainMPS)
    sites1 = _allsites(M1)
    sites2 = _allsites(M2)

    external_sites = setdiff(union(sites1, sites2), intersect(sites1, sites2))

    proj = deepcopy(M1.projector.data)
    empty!(proj)

    for s in external_sites
        if isprojectedat(M1, s)
            proj[s] = M1.projector[s]
        end
        if isprojectedat(M2, s)
            proj[s] = M2.projector[s]
        end
    end

    return Projector(proj), external_sites
end

function _is_externalsites_compatible_with_projector(external_sites, projector)
    for s in keys(projector)
        if !(s ∈ external_sites)
            return false
        end
    end
    return true
end

"""
Project two SubDomainMPS objects to `proj` before contracting them.
"""
function projcontract(
    M1::SubDomainMPS,
    M2::SubDomainMPS,
    proj::Projector;
    alg="zipup",
    cutoff=default_cutoff(),
    maxdim=default_maxdim(),
    verbosity=0,
    kwargs...,
)::Union{Nothing,SubDomainMPS}
    # Project M1 and M2 to `proj` before contracting
    M1 = project(M1, proj)
    M2 = project(M2, proj)
    if M1 === nothing || M2 === nothing
        return nothing
    end

    _, external_sites = _projector_after_contract(M1, M2)

    if !_is_externalsites_compatible_with_projector(external_sites, proj)
        error("The projector contains projection onto a site that is not an external site.")
    end

    # t1 = time_ns()
    r = contract(M1, M2; alg, cutoff, maxdim, kwargs...)
    # t2 = time_ns()
    #println("contract: $((t2 - t1)*1e-9) s")
    return r
end

"""
Project two SubDomainMPS objects to `proj` before contracting them.
The results are summed.
"""
function projcontract(
    M1::AbstractVector{SubDomainMPS},
    M2::AbstractVector{SubDomainMPS},
    proj::Projector;
    alg="zipup",
    alg_sum="fit",
    cutoff=default_cutoff(),
    maxdim=default_maxdim(),
    patchorder=Index[],
    kwargs...,
)::Union{Nothing,Vector{SubDomainMPS}}
    results = SubDomainMPS[]
    #T1 = time_ns()
    for M1_ in M1
        for M2_ in M2
            #t1 = time_ns()
            r = projcontract(M1_, M2_, proj; alg, cutoff, maxdim, kwargs...)
            #t2 = time_ns()
            if r !== nothing
                push!(results, r)
            end
        end
    end
    #T2 = time_ns()
    #println("projcontract, all: $((T2 - T1)*1e-9) s")

    if isempty(results)
        return nothing
    end

    if length(results) == 1
        return results
    end

    res = if length(patchorder) > 0
        _add_patching(results; cutoff, maxdim, patchorder, kwargs...)
    else
        [_add(results...; alg=alg_sum, cutoff, maxdim, kwargs...)]
    end
    #T3 = time_ns()
    #println("mul: $((T2 - T1)*1e-9) s")
    #println("add: $((T3 - T2)*1e-9) s")
    return res
end

"""
Contract two Blocked MPS objects.

At each site, the objects must share at least one site index.
"""
function contract(
    M1::PartitionedMPS,
    M2::PartitionedMPS;
    alg="zipup",
    cutoff=default_cutoff(),
    maxdim=default_maxdim(),
    patchorder=Index[],
    kwargs...,
)::Union{PartitionedMPS}
    M = PartitionedMPS()
    return contract!(M, M1, M2; alg, cutoff, maxdim, patchorder, kwargs...)
end

"""
Contract two PartitionedMPS objects.

Existing blocks `M` in the resulting PartitionedMPS will be overwritten if `overwrite=true`.
"""
function contract!(
    M::PartitionedMPS,
    M1::PartitionedMPS,
    M2::PartitionedMPS;
    alg="zipup",
    cutoff=default_cutoff(),
    maxdim=default_maxdim(),
    patchorder=Index[],
    overwrite=true,
    kwargs...,
)::Union{PartitionedMPS}
    blocks = OrderedSet((
        _projector_after_contract(b1, b2)[1] for b1 in values(M1), b2 in values(M2)
    ))
    for b1 in blocks, b2 in blocks
        if b1 != b2 && hasoverlap(b1, b2)
            error("After contraction, projectors must not overlap.")
        end
    end
    M1_::Vector{SubDomainMPS} = collect(values(M1))
    M2_::Vector{SubDomainMPS} = collect(values(M2))
    for b in blocks
        if haskey(M.data, b) && !overwrite
            continue
        end
        res = projcontract(M1_, M2_, b; alg, cutoff, maxdim, patchorder, kwargs...)
        if res !== nothing
            append!(M, res)
        end
    end
    return M
end
