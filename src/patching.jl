"""
Add multiple SubDomainMPS objects on the same projector.

If the bond dimension of the result reaches `maxdim`,
perform patching recursively to reduce the bond dimension.
"""
function _add_patching(
    subdmpss::AbstractVector{SubDomainMPS};
    cutoff=0.0,
    maxdim=typemax(Int),
    alg="fit",
    patchorder=Index[],
)::Vector{SubDomainMPS}
    if length(unique([sudmps.projector for sudmps in subdmpss])) != 1
        error("All SubDomainMPS objects must have the same projector.")
    end

    # First perform addition upto given maxdim
    # TODO: Early termination if the bond dimension reaches maxdim
    sum_approx = _add(subdmpss...; alg, cutoff, maxdim)

    # If the bond dimension is less than maxdim, return the result
    maxbonddim(sum_approx) < maxdim && return [sum_approx]

    # @assert maxbonddim(sum_approx) == maxdim

    nextprjidx = _next_projindex(subdmpss[1].projector, patchorder)

    nextprjidx === nothing && return [sum_approx]

    blocks = SubDomainMPS[]
    for prjval in 1:ITensors.dim(nextprjidx)
        prj_ = subdmpss[1].projector & Projector(nextprjidx => prjval)
        blocks =
            blocks ∪ _add_patching(
                [project(sudmps, prj_) for sudmps in subdmpss];
                cutoff,
                maxdim,
                alg,
                patchorder,
            )
    end

    return blocks
end

"""
Return the next index to be projected.
"""
function _next_projindex(prj::Projector, patchorder)::Union{Nothing,Index}
    idx = findfirst(x -> !isprojectedat(prj, x), patchorder)
    if idx === nothing
        return nothing
    else
        return patchorder[idx]
    end
end

"""
Add multiple PartitionedMPS objects.
"""
function add_patching(
    partmps::AbstractVector{PartitionedMPS};
    cutoff=0.0,
    maxdim=typemax(Int),
    alg="fit",
    patchorder=Index[],
)::PartitionedMPS
    result = _add_patching(
        union(values(x) for x in partmps); cutoff, maxdim, alg, patchorder
    )
    return PartitionedMPS(result)
end

"""
Adaptive patching

Do patching recursively to reduce the bond dimension.
If the bond dimension of a SubDomainMPS exceeds `maxdim`, perform patching.
"""
function adaptive_patching(
    subdmps::SubDomainMPS, patchorder; cutoff=0.0, maxdim=typemax(Int)
)::Vector{SubDomainMPS}
    prjidx = _next_projindex(subdmps.projector, patchorder)
    if maxbonddim(subdmps) <= maxdim || prjidx === nothing
        return [subdmps]
    end

    # The bond dimension exceeds maxdim and there's an index to be projected, so perform patching
    prjvals = 1:ITensors.dim(prjidx)
    return mapreduce(vcat, prjvals) do prjval
        prj_ = subdmps.projector & Projector(prjidx => prjval)
        subdmps_ = truncate(project(subdmps, prj_); cutoff)
        adaptive_patching(subdmps_, patchorder; cutoff, maxdim)
    end
end

"""
Adaptive patching

Do patching recursively to reduce the bond dimension.
If the bond dimension of a SubDomainMPS exceeds `maxdim`, perform patching.
"""
function adaptive_patching(
    prjmps::PartitionedMPS, patchorder; cutoff=0.0, maxdim=typemax(Int)
)::PartitionedMPS
    return PartitionedMPS(
        collect(
            Iterators.flatten(
                adaptive_patching(subdmps, patchorder; cutoff, maxdim) for
                subdmps in values(prjmps)
            ),
        ),
    )
end
