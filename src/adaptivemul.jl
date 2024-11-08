"""
Lazy evaluation for contraction of two ProjMPS objects.
"""
struct LazyContraction
    a::ProjMPS
    b::ProjMPS
    projector::Projector # Projector for the external indices of (a * b)
    function LazyContraction(a::ProjMPS, b::ProjMPS)
        shared_inds = Set{Index}()
        for (a_, b_) in zip(siteinds(a), siteinds(b))
            cinds = commoninds(a_, b_)
            length(cinds) > 0 || error("The two ProjMPS must have common indices at every site.")
            shared_inds = shared_inds ∪ cinds
        end
        #@show  typeof(_projector_after_contract(a, b))
        new(a, b, _projector_after_contract(a, b)[1])
    end
end

function lazycontraction(a::ProjMPS, b::ProjMPS)::Union{LazyContraction, Nothing}
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
Perform contruction of two BlockedMPS objects.

The ProjMPS objects of each BlockedMPS do not overlap with each other.
This makes the algorithm much simpler
"""
function adaptivecontract(
    a::BlockedMPS,
    b::BlockedMPS,
    pordering::AbstractVector{Index} = Index[];
    alg = ITensors.Algorithm"fit"(),
    cutoff = 1e-25,
    maxdim = typemax(Int),
    kwargs...
)
    patches = Dict{Projector, Vector{Union{ProjMPS,LazyContraction}}}()

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
    # This should be prohibited by the fact that the blocks in each ProjMPS obejct do not overlap.
    isdisjoint(collect(keys(patches))) || error("Overlapping projectors")

    result_blocks = ProjMPS[]
    for (p, muls) in patches
        prjmpss = [contract(m.a, m.b; alg, cutoff, maxdim, kwargs...) for m in muls]
        #patches[p] = +(prjmpss...; alg="fit", cutoff, maxdim)
        push!(result_blocks,  +(prjmpss...; alg="fit", cutoff, maxdim))
    end

    return BlockedMPS(result_blocks)
end
