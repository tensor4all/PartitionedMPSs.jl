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
    new_a = project(obj.a, a.projector & prj; kwargs...)
    new_b = project(obj.b, b.projector & prj; kwargs...)
    return LazyContraction(new_a, new_b)
end

"""
Perform contruction of two PartitionedMPS objects.

The SubDomainMPS objects of each PartitionedMPS do not overlap with each other.
This makes the algorithm much simpler
"""
function adaptivecontract(
    a::PartitionedMPS,
    b::PartitionedMPS,
    pordering::AbstractVector{Index}=Index[];
    alg="fit",
    cutoff=default_cutoff(),
    maxdim=default_maxdim(),
    kwargs...,
)
    patches = Dict{Projector,Vector{Union{SubDomainMPS,LazyContraction}}}()

    for x in values(a), y in values(b) # FIXME: Naive loop over O(N^2) pairs
        xy = lazycontraction(x, y)
        if xy === nothing
            continue
        end
        if haskey(patches, xy.projector)
            push!(patches[xy.projector], xy)
        else
            patches[xy.projector] = [xy]
        end
    end

    # Check no overlapping projectors.
    # This should be prohibited by the fact that the blocks in each SubDomainMPS obejct do not overlap.
    isdisjoint(collect(keys(patches))) || error("Overlapping projectors")

    result_blocks = SubDomainMPS[]
    for (p, muls) in patches
        subdmps = [contract(m.a, m.b; alg, cutoff, maxdim, kwargs...) for m in muls]
        #patches[p] = +(subdmps...; alg="fit", cutoff, maxdim)
        push!(result_blocks, +(subdmps...; alg="fit", cutoff, maxdim))
    end

    return PartitionedMPS(result_blocks)
end
