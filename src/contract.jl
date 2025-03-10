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

    # Precollect the pairs for threading
    pairinfo = vec([(m1, m2, maxlinkdim(m1) * maxlinkdim(m2)) for m1 in M1, m2 in M2])
    # Heavy contraction first
    sort!(pairinfo; by=x -> x[3], rev=true)
    # Lock for threaded computation 
    local_lock = ReentrantLock()

    if Threads.nthreads() > 1
        nT = nthreads()
        chunked_pairs = [Vector{Tuple{SubDomainMPS,SubDomainMPS}}() for _ in 1:nT]
        # Equally divide expensive computations btw threads
        for (i, (m1, m2, _)) in enumerate(pairinfo)
            t = ((i - 1) % nT) + 1
            push!(chunked_pairs[t], (m1, m2))
        end

        @threads for t in 1:nT
            local_buffer = SubDomainMPS[]

            for (m1, m2) in chunked_pairs[t]
                r = projcontract(m1, m2, proj; alg, cutoff, maxdim, kwargs...)

                if r !== nothing
                    push!(local_buffer, r) # Thread-local accumulation
                end
            end

            # Lock is held only briefly to merge partial results
            lock(local_lock) do
                append!(results, local_buffer)
            end
        end
    else
        for (m1, m2, _) in pairinfo
            r = projcontract(m1, m2, proj; alg, cutoff, maxdim, kwargs...)
            if r !== nothing
                push!(results, r)
            end
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
    parallel::Symbol=:serial,
    kwargs...,
)::Union{PartitionedMPS}
    M = PartitionedMPS()
    if parallel == :distributed_thread
        return parallel_contract!(M, M1, M2; alg, cutoff, maxdim, patchorder, kwargs...)
    elseif parallel == :distributed
        return distribute_contract!(M, M1, M2; alg, cutoff, maxdim, patchorder, kwargs...)
    elseif parallel == :serial
        return contract!(M, M1, M2; alg, cutoff, maxdim, patchorder, kwargs...)
    else
        error("Symbol $(parallel) not recongnized.")
    end
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
    blocks_to_sets = Dict{Projector,Tuple{Set{SubDomainMPS},Set{SubDomainMPS}}}()

    for m1 in values(M1), m2 in values(M2)
        if hasoverlap(m1.projector, m2.projector)
            block = add_entry!(blocks_to_sets, _projector_after_contract(m1, m2)[1])
            if haskey(blocks_to_sets, block)
                set1, set2 = blocks_to_sets[block]
                push!(set1, m1)
                push!(set2, m2)
            else
                blocks_to_sets[block] = (Set([m1]), Set([m2]))
            end
        end
    end

    for b1 in keys(blocks_to_sets), b2 in keys(blocks_to_sets)
        if b1 != b2 && hasoverlap(b1, b2)
            error("After contraction, projectors must not overlap.")
        end
    end

    # Builds tasks to parallelise
    tasks = Vector{Tuple{Projector,Vector{SubDomainMPS},Vector{SubDomainMPS}}}()
    for (proj, (set1, set2)) in blocks_to_sets
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

function parallel_contract!(
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
    blocks_to_sets = Dict{Projector,Tuple{Set{SubDomainMPS},Set{SubDomainMPS}}}()

    for m1 in values(M1), m2 in values(M2)
        if hasoverlap(m1.projector, m2.projector)
            block = add_entry!(blocks_to_sets, _projector_after_contract(m1, m2)[1])
            if haskey(blocks_to_sets, block)
                set1, set2 = blocks_to_sets[block]
                push!(set1, m1)
                push!(set2, m2)
            else
                blocks_to_sets[block] = (Set([m1]), Set([m2]))
            end
        end
    end

    for b1 in keys(blocks_to_sets), b2 in keys(blocks_to_sets)
        if b1 != b2 && hasoverlap(b1, b2)
            error("After contraction, projectors must not overlap.")
        end
    end

    # Builds tasks to parallelise
    tasks = Vector{Tuple{Projector,Vector{SubDomainMPS},Vector{SubDomainMPS}}}()
    for (proj, (set1, set2)) in blocks_to_sets
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

    results_parallel = pmap(process_task, tasks)

    for res in results_parallel
        if res !== nothing
            append!(M, res)
        end
    end

    return M
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
    blocks_to_sets = Dict{Projector,Tuple{Set{SubDomainMPS},Set{SubDomainMPS}}}()

    for m1 in values(M1), m2 in values(M2)
        if hasoverlap(m1.projector, m2.projector)
            block = add_entry!(blocks_to_sets, _projector_after_contract(m1, m2)[1])
            if haskey(blocks_to_sets, block)
                set1, set2 = blocks_to_sets[block]
                push!(set1, m1)
                push!(set2, m2)
            else
                blocks_to_sets[block] = (Set([m1]), Set([m2]))
            end
        end
    end

    for b1 in keys(blocks_to_sets), b2 in keys(blocks_to_sets)
        if b1 != b2 && hasoverlap(b1, b2)
            error("After contraction, projectors must not overlap.")
        end
    end

    tasks = Vector{Tuple{Projector,SubDomainMPS,SubDomainMPS}}()
    for (proj, (set1, set2)) in blocks_to_sets
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

    block_group = Dict{Projector,Vector{SubDomainMPS}}()
    for (b, subdmps) in valid_results
        if haskey(block_group, b)
            push!(block_group[b], subdmps)
        else
            block_group[b] = [subdmps]
        end
    end

    block_group_array = collect(block_group)

    function sum_blocks(group; patchorder, alg_sum, cutoff, maxdim, kwargs...)
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
        group -> sum_blocks(
            group;
            patchorder=patchorder,
            alg_sum=alg_sum,
            cutoff=cutoff,
            maxdim=maxdim,
            kwargs...,
        ),
        block_group_array,
    )

    for res in summed_patches
        if res !== nothing
            append!(M, vcat(res))
        end
    end

    return M
end
