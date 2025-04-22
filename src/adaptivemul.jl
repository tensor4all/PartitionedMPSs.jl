"""
Lazy evaluation for contraction of two SubDomainMPS objects.
"""
struct LazyContraction
    a::SubDomainMPS
    b::SubDomainMPS
    projector::Projector # Projector for the external indices of (a * b)
    function LazyContraction(a::SubDomainMPS, b::SubDomainMPS)
        shared_inds = Set{Index}()
        for (a_, b_) in zip(siteinds(a), siteinds(b))
            cinds = commoninds(a_, b_)
            length(cinds) > 0 ||
                error("The two SubDomainMPS must have common indices at every site.")
            shared_inds = shared_inds ∪ cinds
        end
        #@show  typeof(_projector_after_contract(a, b))
        return new(a, b, _projector_after_contract(a, b)[1])
    end
end

function lazycontraction(a::SubDomainMPS, b::SubDomainMPS)::Union{LazyContraction,Nothing}
    # If any of shared indices between a and b is projected at different levels, return nothing
    if a.projector & b.projector === nothing
        return nothing
    end
    return LazyContraction(a, b)
end

Base.length(obj::LazyContraction) = length(obj.a)

"""
Project the LazyContraction object to `prj` before evaluating it.

This may result in projecting the external indices of `a` and `b`.
"""
function project(obj::LazyContraction, prj::Projector; kwargs...)::LazyContraction
    new_a = project(obj.a, prj; kwargs...)
    new_b = project(obj.b, prj; kwargs...)
    if isnothing(new_a) || isnothing(new_b)
        error("New projector is not compatible with SubDomainMPSs projectors.")
    end
    return LazyContraction(new_a, new_b)
end

# Preprocessing of the patches to obtain all the contraction tasks from two PartitionedMPSs
function _adaptivecontraction_tasks(
    M1::PartitionedMPS, M2::PartitionedMPS
)::Dict{Projector,Vector{Union{SubDomainMPS,LazyContraction}}}
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

    # Transform the SubDomainMPS pairs in LazyContraction wrappers
    tasks = Dict{Projector,Vector{Union{SubDomainMPS,LazyContraction}}}()

    for (proj, subdmps_pair) in final_patches
        resultvec = Union{SubDomainMPS,LazyContraction}[]
        for pair in subdmps_pair
            # Trim projectors to produce only non-overlapping patches
            lc = project(lazycontraction(pair...), proj)
            if lc === nothing
                @warn "LazyContraction == nothing. Faulty patch preprocessing..." proj pair
            else
                push!(resultvec, lc)
            end
        end
        tasks[proj] = resultvec
    end

    return tasks
end

# Performs the patched contraction of two PartitionedMPS
# For each compatible pair of patches the contraction is attempted and split in smaller patches
# if the result exceeds the fixed bond dimension. 
function patch_contract!(
    patches::Dict{Projector,Vector{Union{SubDomainMPS,LazyContraction}}},
    pordering::AbstractVector{Index{IndsT}},
    maxdim,
    cutoff;
    alg="fit",
    kwargs...,
) where {IndsT}
    # A small helper
    has_lazy() = any(any(lc -> lc isa LazyContraction, v) for v in values(patches))

    # Keep iterating until no LazyContraction remains
    while has_lazy()
        for prj in collect(keys(patches))
            blockvec = patches[prj]
            i = 1
            while i <= length(blockvec)
                m = blockvec[i]
                if m isa LazyContraction
                    # Attempt the actual contraction
                    contracted = contract(
                        m.a, m.b, ; alg=alg, cutoff=cutoff, maxdim=maxdim, kwargs...
                    )
                    isnothing(contracted) && error(
                        "Some contractions failed. Double check the patch ordering..."
                    )

                    # Check the bond dimension of result 
                    max_bdim = maxbonddim(contracted)
                    if max_bdim < maxdim
                        # Good: replace the lazy contraction with final SubDomainMPS
                        blockvec[i] = contracted
                    else
                        # Too large => we must expand the projector
                        nextprjidx = _next_projindex(m.projector, pordering)
                        if nextprjidx === nothing
                            @warn(
                                "Cannot expand further; bond dimension still exceeds maxdim."
                            )
                            # Keep it anyway
                            blockvec[i] = contracted
                            i += 1
                            continue
                        else
                            popat!(blockvec, i)
                            d = ITensors.dim(nextprjidx)
                            for val in 1:d
                                # Construct a new projector that includes (nextprjidx => val)
                                new_prj = m.projector & Projector(nextprjidx => val)

                                new_m = project(m, new_prj)
                                # Add a new lazy contraction to patches[new_prj]
                                push!(
                                    get!(
                                        () -> Vector{Union{SubDomainMPS,LazyContraction}}(),
                                        patches,
                                        new_prj,
                                    ),
                                    new_m,
                                )
                            end
                            # don't increment i, because we removed the old lazy contraction
                            continue
                        end
                    end
                end
                i += 1
            end

            # If the current patch ended up with an empty vector, remove it
            if isempty(blockvec)
                delete!(patches, prj)
            end
        end
    end

    @assert !has_lazy() "Some LazyContraction are still present. Something went wrong..."

    # Check that the final projectors are not overlapping
    return isdisjoint(collect(keys(patches))) || error("Overlapping projectors")
end

"""
Perform contraction of two PartitionedMPS objects.

The resulting patches after the contraction are patch-added if projected on the same final patch. 

"""
function adaptivecontract(
    a::PartitionedMPS,
    b::PartitionedMPS,
    pordering::AbstractVector{Index{IndsT}}=Index{IndsT}[];
    alg="fit",
    alg_sum="fit",
    cutoff=default_cutoff(),
    maxdim=default_maxdim(),
    kwargs...,
) where {IndsT}
    patches = _adaptivecontraction_tasks(a, b)

    # Check no overlapping projectors.
    isdisjoint(collect(keys(patches))) || error("Overlapping projectors")

    # Perform the iterative patch contraction
    patch_contract!(patches, pordering, maxdim, cutoff; alg=alg, kwargs...)

    # Resum SubDomainMPS on the same patch
    result_blocks = SubDomainMPS[]
    for (prj, blockvec) in patches
        # Each entry in blockvec is now guaranteed to be a SubDomainMPS
        subdmps_list = Vector{SubDomainMPS}(blockvec)

        if length(subdmps_list) == 1
            push!(result_blocks, subdmps_list[1])
        else
            patch_sum = _add_patching(
                subdmps_list;
                alg=alg_sum,
                cutoff=cutoff,
                maxdim=maxdim,
                patchorder=pordering,
            )

            append!(result_blocks, patch_sum)
        end
    end

    return PartitionedMPS(result_blocks)
end
