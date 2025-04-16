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
The results are summed or patch-summed if belonging to the same patch.
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

# Function to add a new patch to the result patched contraction. Only if the patch is non-overlapping with any 
# of the already present ones it is added, otherwise it is fused. 
function add_result_patch!(
    dict::Dict{Projector,Vector{Tuple{SubDomainMPS,SubDomainMPS}}}, proj::Projector
)
    # Iterate over a copy of keys (patches) to avoid modifications while looping.
    for existing_proj in collect(keys(dict))
        if hasoverlap(existing_proj, proj)
            fused_proj = existing_proj | proj
            # Save the subdmpss of the overlapping patch.
            subdmpss = dict[existing_proj]
            # Remove the old projector (this deletes also its associated subdmpss).
            delete!(dict, existing_proj)
            # Recursively update with the fused projector.
            new_proj = add_result_patch!(dict, fused_proj)
            # If new_proj is already present, merge the subdmpss; otherwise, insert the saved subdmpss.
            if haskey(dict, new_proj)
                append!(dict[new_proj], subdmpss)
            else
                dict[new_proj] = subdmpss
            end
            return new_proj
        end
    end
    # If no overlapping proj is found, then ensure proj is in the dictionary (sanity passage). 
    if !haskey(dict, proj)
        dict[proj] = Vector{Tuple{SubDomainMPS,SubDomainMPS}}()
    end
    return proj
end

# Preprocessing of the patches to obtain all the contraction tasks from two PartitionedMPSs
function _contraction_tasks(
    M1::PartitionedMPS,
    M2::PartitionedMPS;
    M::PartitionedMPS=PartitionedMPS(),
    overwrite=true,
)::Vector{Tuple{Projector,SubDomainMPS,SubDomainMPS}}
    final_patches = Dict{Projector,Vector{Tuple{SubDomainMPS,SubDomainMPS}}}()
    # Add a new patch only if the two subdmps are compatible (overlapping internal projected
    # sites) and the new patch is non-overlapping with all the existing ones.
    for m1 in values(M1), m2 in values(M2)
        tmp_prj = _projector_after_contract(m1, m2)[1]
        if tmp_prj !== nothing
            patch = add_result_patch!(final_patches, tmp_prj)
            if haskey(final_patches, patch)
                push!(final_patches[patch], (m1, m2))
            else
                final_patches[patch] = (m1, m2)
            end
        end
    end

    # Sanity check
    for p1 in keys(final_patches), p2 in keys(final_patches)
        if p1 != p2 && hasoverlap(p1, p2)
            error("After contraction, projectors must not overlap.")
        end
    end

    # Flatten the result to create contraction tasks
    tasks = Vector{Tuple{Projector,SubDomainMPS,SubDomainMPS}}()
    for (proj, submps_pairs) in final_patches
        if haskey(M.data, proj) && !overwrite
            continue
        end
        for (subdmps1, subdmps2) in submps_pairs
            push!(tasks, (proj, subdmps1, subdmps2))
        end
    end

    return tasks
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
    return contract!(M, M1, M2; alg, cutoff, maxdim, patchorder, parallel, kwargs...)
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
    alg_sum="fit",
    cutoff=default_cutoff(),
    maxdim=default_maxdim(),
    patchorder=Index[],
    parallel::Symbol=:serial,
    overwrite=true,
    kwargs...,
)::PartitionedMPS
    # Builds contraction tasks 
    tasks = _contraction_tasks(M1, M2; M=M, overwrite=overwrite)

    # Helper contraction function
    function contract_task(task; alg, cutoff, maxdim, kwargs...)
        proj, M1_subs, M2_subs = task
        return projcontract(M1_subs, M2_subs, proj; alg, cutoff, maxdim, kwargs...)
    end

    # Serial or distributed contraction
    if parallel == :serial
        contr_results = map(
            task -> contract_task(task; alg, cutoff, maxdim, kwargs...), tasks
        )
    elseif parallel == :distributed
        contr_results = pmap(
            task -> contract_task(task; alg, cutoff, maxdim, kwargs...), tasks
        )
    else
        error("Symbol $(parallel) not recongnized.")
    end

    # Sanity check
    all(r -> r !== nothing, contr_results) ||
        error("Some contraction returned `nothing`. Faulty preprocessing of patches...")

    ## Resum SubDomainMPSs projected on the same final patch
    # Group together patches to resum 
    patch_group = Dict{Projector,Vector{SubDomainMPS}}()
    for subdmps in contr_results
        if haskey(patch_group, subdmps.projector)
            push!(patch_group[subdmps.projector], subdmps)
        else
            patch_group[subdmps.projector] = [subdmps]
        end
    end

    # Helper sum function
    function sum_task(group; patchorder, alg_sum, cutoff, maxdim, kwargs...)
        if length(group) == 1
            return [group[1]]
        else
            res = if length(patchorder) > 0
                _add_patching(group; cutoff, maxdim, patchorder, kwargs...)
            else
                [_add(group...; alg=alg_sum, cutoff, maxdim, kwargs...)]
            end
            return res
        end
    end

    if parallel == :serial
        summed_patches = map(
            group -> sum_task(
                group;
                patchorder=patchorder,
                alg_sum=alg_sum,
                cutoff=cutoff,
                maxdim=maxdim,
                kwargs...,
            ),
            collect(values(patch_group)),
        )
    elseif parallel == :distributed
        summed_patches = pmap(
            group -> sum_task(
                group;
                patchorder=patchorder,
                alg_sum=alg_sum,
                cutoff=cutoff,
                maxdim=maxdim,
                kwargs...,
            ),
            collect(values(patch_group)),
        )
    end

    # Assembling the PartitionedMPS
    for subdmpss in summed_patches
        if subdmpss !== nothing
            append!(M, vcat(subdmpss))
        end
    end

    return M
end
