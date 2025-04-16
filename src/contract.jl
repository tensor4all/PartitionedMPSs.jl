# just for backward compatibility...
_alg_map = Dict(
    ITensors.Algorithm(alg) => alg for alg in ["directsum", "densitymatrix", "fit", "naive"]
)
""" 
Contraction of two SubDomainMPSs. 
Only if the shared projected indices overlap the contraction is non-vanishing.
"""
function contract(
    M1::SubDomainMPS, M2::SubDomainMPS; alg, kwargs...
)::Union{SubDomainMPS,Nothing}
    # If the SubDomainMPS don't overlap they cannot be contracted.
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
    # If the SubDomainMPS don't overlap they cannot be contracted -> no final projector
    if !hasoverlap(M1.projector, M2.projector)
        return nothing, external_sites
    end

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

# Check for newly projected sites to be only external sites.
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

    r = contract(M1, M2; alg, cutoff, maxdim, kwargs...)
    return r
end

"""
Project SubDomainMPS vectors to `proj` before computing all possible pairwise contractions of the elements.
The results are summed or patch-summed.
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

    for m1 in M1, m2 in M2
        r = projcontract(m1, m2, proj; alg, cutoff, maxdim, kwargs...)
        if r !== nothing
            push!(results, r)
        end
    end

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

    return res
end

"""
Contract two PartitionedMPSs MPS objects.

At each site, the objects must share at least one site index.
"""
function contract(
    M1::PartitionedMPS,
    M2::PartitionedMPS;
    alg="zipup",
    cutoff=default_cutoff(),
    maxdim=default_maxdim(),
    patchorder=Index[],
    parallel::Symbol=:serial,
    kwargs...,
)::Union{PartitionedMPS}
    M = PartitionedMPS()
    if parallel == :distributed
        return distribute_contract!(M, M1, M2; alg, cutoff, maxdim, patchorder, kwargs...)
    elseif parallel == :serial
        return contract!(M, M1, M2; alg, cutoff, maxdim, patchorder, kwargs...)
    else
        error("Symbol $(parallel) not recongnized.")
    end
end

function add_entry!(
    dict::Dict{Projector,Tuple{Set{SubDomainMPS},Set{SubDomainMPS}}}, proj::Projector
)
    # Iterate over a copy of keys to avoid modifying the dict while looping.
    for existing in collect(keys(dict))
        if hasoverlap(existing, proj)
            fused_proj = existing | proj
            # Save the current value for the overlapping key.
            val = dict[existing]
            # Remove the old key (this deletes its associated value).
            delete!(dict, existing)
            # Recursively update with the fused projector.
            new_key = add_entry!(dict, fused_proj)
            # If new_key is already present, merge the values; otherwise, insert the saved value.
            if haskey(dict, new_key)
                old_val = dict[new_key]
                dict[new_key] = (union(old_val[1], val[1]), union(old_val[2], val[2]))
            else
                dict[new_key] = val
            end
            return new_key
        end
    end
    # If no overlapping key is found, then ensure proj is in the dictionary.
    if !haskey(dict, proj)
        dict[proj] = (Set{SubDomainMPS}(), Set{SubDomainMPS}())
    end
    return proj
end

"""
Contract two PartitionedMPS objects.

Existing patches `M` in the resulting PartitionedMPS will be overwritten if `overwrite=true`.
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
    patches_to_sets = Dict{Projector,Tuple{Set{SubDomainMPS},Set{SubDomainMPS}}}()

    for m1 in values(M1), m2 in values(M2)
        if hasoverlap(m1.projector, m2.projector)
            patch = add_entry!(patches_to_sets, _projector_after_contract(m1, m2)[1])
            if haskey(patches_to_sets, patch)
                set1, set2 = patches_to_sets[patch]
                push!(set1, m1)
                push!(set2, m2)
            else
                patches_to_sets[patch] = (Set([m1]), Set([m2]))
            end
        end
    end

    for b1 in keys(patches_to_sets), b2 in keys(patches_to_sets)
        if b1 != b2 && hasoverlap(b1, b2)
            error("After contraction, projectors must not overlap.")
        end
    end

    # Builds tasks to parallelise
    tasks = Vector{Tuple{Projector,Vector{SubDomainMPS},Vector{SubDomainMPS}}}()
    for (proj, (set1, set2)) in patches_to_sets
        if haskey(M.data, proj) && !overwrite
            continue
        end
        push!(tasks, (proj, collect(set1), collect(set2)))
    end

    function process_task(task)
        proj, M1_subs, M2_subs = task
        return projcontract(
            M1_subs, M2_subs, proj; alg, cutoff, maxdim, patchorder, kwargs...
        )
    end

    results = map(process_task, tasks)

    for res in results
        if res !== nothing
            append!(M, res)
        end
    end

    return M
end

function distribute_contract!(
    M::PartitionedMPS,
    M1::PartitionedMPS,
    M2::PartitionedMPS;
    alg="zipup",
    alg_sum="fit",
    cutoff=default_cutoff(),
    maxdim=default_maxdim(),
    patchorder=Index[],
    overwrite=true,
    kwargs...,
)::Union{PartitionedMPS}
    patches_to_sets = Dict{Projector,Tuple{Set{SubDomainMPS},Set{SubDomainMPS}}}()

    for m1 in values(M1), m2 in values(M2)
        if hasoverlap(m1.projector, m2.projector)
            patch = add_entry!(patches_to_sets, _projector_after_contract(m1, m2)[1])
            if haskey(patches_to_sets, patch)
                set1, set2 = patches_to_sets[patch]
                push!(set1, m1)
                push!(set2, m2)
            else
                patches_to_sets[patch] = (Set([m1]), Set([m2]))
            end
        end
    end

    for b1 in keys(patches_to_sets), b2 in keys(patches_to_sets)
        if b1 != b2 && hasoverlap(b1, b2)
            error("After contraction, projectors must not overlap.")
        end
    end

    tasks = Vector{Tuple{Projector,SubDomainMPS,SubDomainMPS}}()
    for (proj, (set1, set2)) in patches_to_sets
        for subdmps1 in set1, subdmps2 in set2
            if haskey(M.data, proj) && !overwrite
                continue
            end
            push!(tasks, (proj, subdmps1, subdmps2))
        end
    end

    function process_task(task_tuple; alg, cutoff, maxdim, kwargs...)
        # Unpack the tuple
        proj, subdmps1, subdmps2 = task_tuple
        res = projcontract(subdmps1, subdmps2, proj; alg, cutoff, maxdim, kwargs...)
        return (proj, res)
    end

    results = pmap(task -> process_task(task; alg, cutoff, maxdim, kwargs...), tasks)
    valid_results = filter(x -> x[2] !== nothing, results)

    patch_group = Dict{Projector,Vector{SubDomainMPS}}()
    for (b, subdmps) in valid_results
        if haskey(patch_group, b)
            push!(patch_group[b], subdmps)
        else
            patch_group[b] = [subdmps]
        end
    end

    patch_group_array = collect(patch_group)

    function sum_patches(group; patchorder, alg_sum, cutoff, maxdim, kwargs...)
        b, subdmps_list = group
        if length(subdmps_list) == 1
            return [subdmps_list[1]]
        else
            res = if length(patchorder) > 0
                _add_patching(subdmps_list; cutoff, maxdim, patchorder, kwargs...)
            else
                [_add(subdmps_list...; alg=alg_sum, cutoff, maxdim, kwargs...)]
            end
            return res
        end
    end

    summed_patches = pmap(
        group -> sum_patches(
            group;
            patchorder=patchorder,
            alg_sum=alg_sum,
            cutoff=cutoff,
            maxdim=maxdim,
            kwargs...,
        ),
        patch_group_array,
    )

    for res in summed_patches
        if res !== nothing
            append!(M, vcat(res))
        end
    end

    return M
end
