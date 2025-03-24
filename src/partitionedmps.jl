
"""
PartitionedMPS is a structure that holds multiple MPSs (SubDomainMPS) that are associated with different non-overlapping projectors.
"""
struct PartitionedMPS
    data::OrderedDict{Projector,SubDomainMPS}

    function PartitionedMPS(data::AbstractVector{SubDomainMPS})
        sites_all = [siteinds(subdmps) for subdmps in data]
        for n in 2:length(data)
            Set(sites_all[n]) == Set(sites_all[1]) || error("Sitedims mismatch")
        end
        isdisjoint([subdmps.projector for subdmps in data]) || error("Projectors are overlapping")

        dict_ = OrderedDict{Projector,SubDomainMPS}(
            data[i].projector => data[i] for i in 1:length(data)
        )
        return new(dict_)
    end
end

PartitionedMPS(data::SubDomainMPS) = PartitionedMPS([data])

PartitionedMPS() = PartitionedMPS(SubDomainMPS[])

projectors(obj::PartitionedMPS) = keys(obj)

function Base.append!(a::PartitionedMPS, b::PartitionedMPS)
    if !isdisjoint(collect(union(projectors(a), projectors(b))))
        error("Projectors are overlapping")
    end
    for (k, v) in b.data
        a.data[k] = v
    end
    return a
end

function Base.append!(a::PartitionedMPS, b::AbstractVector{SubDomainMPS})
    return append!(a, PartitionedMPS(b))
end

"""
Return the site indices of the PartitionedMPS.
The site indices are returned as a vector of sets, where each set corresponds to the site indices at each site.
"""
function siteindices(obj::PartitionedMPS)
    return [Set(x) for x in ITensors.siteinds(first(values(obj.data)))]
end

ITensors.siteinds(obj::PartitionedMPS) = siteindices(obj)

"""
Get the number of the data in the PartitionedMPS.
This is NOT the number of sites in the PartitionedMPS.
"""
Base.length(obj::PartitionedMPS) = length(obj.data)

"""
Indexing for PartitionedMPS. This is deprecated and will be removed in the future.
"""
function Base.getindex(partmps::PartitionedMPS, i::Integer)::SubDomainMPS
    @warn "Indexing for PartitionedMPS is deprecated. Use getindex(partmps, p::Projector) instead."
    return first(Iterators.drop(values(partmps.data), i - 1))
end

Base.getindex(obj::PartitionedMPS, p::Projector) = obj.data[p]

function Base.iterate(partmps::PartitionedMPS, state)
    return iterate(partmps.data, state)
end

function Base.iterate(partmps::PartitionedMPS)
    return iterate(partmps.data)
end

"""
Return the keys, i.e., projectors of the PartitionedMPS.
"""
function Base.keys(obj::PartitionedMPS)
    return keys(obj.data)
end

"""
Return the values, i.e., SubDomainMPS of the PartitionedMPS.
"""
function Base.values(obj::PartitionedMPS)
    return values(obj.data)
end

"""
Rearrange the site indices of the PartitionedMPS according to the given order.
If nessecary, tensors are fused or split to match the new order.
"""
function rearrange_siteinds(obj::PartitionedMPS, sites)
    return PartitionedMPS([rearrange_siteinds(subdmps, sites) for subdmps in values(obj)])
end

function prime(Ψ::PartitionedMPS, args...; kwargs...)
    return PartitionedMPS([
        prime(subdmps, args...; kwargs...) for subdmps in values(Ψ.data)
    ])
end

function noprime(Ψ::PartitionedMPS, args...; kwargs...)
    return PartitionedMPS([
        noprime(subdmps, args...; kwargs...) for subdmps in values(Ψ.data)
    ])
end

"""
Return the norm of the PartitionedMPS.
"""
function LinearAlgebra.norm(M::PartitionedMPS)
    return sqrt(reduce(+, (x^2 for x in LinearAlgebra.norm.(values(M)))))
end

"""
Add two PartitionedMPS objects.

If the two projects have the same projectors in the same order, the resulting PartitionedMPS will have the same projectors in the same order.
By default, we use `directsum` algorithm to compute the sum and no truncation is performed.
"""
function Base.:+(
    partmpss::PartitionedMPS...;
    alg="directsum",
    cutoff=0.0,
    maxdim=typemax(Int),
    coeffs=ones(length(partmpss)),
    kwargs...,
)::PartitionedMPS
    result = PartitionedMPS()
    for (coeff, partmps) in zip(coeffs, partmpss)
        result = +(result, coeff * partmps; alg, cutoff, maxdim, kwargs...)
    end
    return result
end

function Base.:+(
    a::PartitionedMPS,
    b::PartitionedMPS;
    alg="directsum",
    cutoff=0.0,
    maxdim=typemax(Int),
    coeffs=(1.0, 1.0),
    kwargs...,
)::PartitionedMPS
    result = PartitionedMPS()
    return add!(result, a, b; alg, cutoff, maxdim, coeffs, kwargs...)
end

function add!(
    result::PartitionedMPS,
    a::PartitionedMPS,
    b::PartitionedMPS;
    alg="directsum",
    cutoff=0.0,
    maxdim=typemax(Int),
    overwrite=true,
    coeffs=(1.0, 1.0),
    kwargs...,
)::PartitionedMPS
    length(coeffs) == 2 || error("coeffs must be a tuple of length 2")
    data = SubDomainMPS[]
    for k in unique(vcat(collect(keys(a)), collect(keys(b)))) # preserve order
        if k ∈ keys(result) && !overwrite
            continue
        end
        if k ∈ keys(a) && k ∈ keys(b)
            a[k].projector == b[k].projector || error("Projectors mismatch at $(k)")
            push!(
                data, +(coeffs[1] * a[k], coeffs[2] * b[k]; alg, cutoff, maxdim, kwargs...)
            )
        elseif k ∈ keys(a)
            push!(data, coeffs[1] * a[k])
        elseif k ∈ keys(b)
            push!(data, coeffs[2] * b[k])
        else
            error("Something went wrong")
        end
    end
    return append!(result, data)
end

function Base.:*(a::PartitionedMPS, b::Number)::PartitionedMPS
    return PartitionedMPS([a[k] * b for k in keys(a)])
end

function Base.:*(a::Number, b::PartitionedMPS)::PartitionedMPS
    return b * a
end

function Base.:-(obj::PartitionedMPS)::PartitionedMPS
    return -1 * obj
end

"""
Truncate a PartitionedMPS object piecewise.

Each SubDomainMPS in the PartitionedMPS is truncated independently,
but the cutoff is adjusted according to the norm of each SubDomainMPS.
The total error is the sum of the errors in each SubDomainMPS.
"""
function truncate(
    obj::PartitionedMPS;
    cutoff=default_cutoff(),
    maxdim=default_maxdim(),
    use_adaptive_weight=true,
    maxrefinement=4,
    kwargs...,
)::PartitionedMPS
    norm2 = [LinearAlgebra.norm(v)^2 for v in values(obj)]
    total_norm2 = sum(norm2)
    weights = [total_norm2 / norm2_v for norm2_v in norm2] # Initial weights (FIXME: better choice?)

    compressed = obj

    for _ in 1:maxrefinement
        compressed = PartitionedMPS([
            truncate(v; cutoff=cutoff * w, maxdim, kwargs...) for
            (v, w) in zip(values(obj), weights)
        ])
        actual_error = dist(obj, compressed)^2 / sum(norm2)
        if actual_error < cutoff || !use_adaptive_weight
            break
        end

        weights .*= min(cutoff / actual_error, 0.5) # Adjust weights
    end

    return compressed
end

# Only for debug
function ITensorMPS.MPS(
    obj::PartitionedMPS; cutoff=default_cutoff(), maxdim=default_maxdim()
)::MPS
    return reduce(
        (x, y) -> truncate(+(x, y; alg="directsum"); cutoff, maxdim), values(obj.data)
    ).data # direct sum
end

function ITensorMPS.MPO(
    obj::PartitionedMPS; cutoff=default_cutoff(), maxdim=default_maxdim()
)::MPO
    return MPO(collect(MPS(obj; cutoff=cutoff, maxdim=maxdim)))
end

"""
Make the PartitionedMPS diagonal for a given site index `s` by introducing a dummy index `s'`.
"""
function makesitediagonal(
    obj::PartitionedMPS, sites::AbstractVector{Index{IndsT}}; baseplev=0
) where {IndsT}
    return PartitionedMPS([
        makesitediagonal(subdmps, sites; baseplev=baseplev) for subdmps in values(obj)
    ])
end

function makesitediagonal(obj::PartitionedMPS, site::Index{IndsT}; baseplev=0) where {IndsT}
    return PartitionedMPS([
        makesitediagonal(subdmps, site; baseplev=baseplev) for subdmps in values(obj)
    ])
end

"""
Extract diagonal of the PartitionedMPS for `s`, `s'`, ... for a given site index `s`,
where `s` must have a prime level of 0.
"""
function extractdiagonal(obj::PartitionedMPS, sites)
    return PartitionedMPS([extractdiagonal(subdmps, sites) for subdmps in values(obj)])
end

function dist(a::PartitionedMPS, b::PartitionedMPS)
    return sqrt(sum(ITensorMPS.dist(MPS(a[k]), MPS(b[k]))^2 for k in keys(a)))
end

function _findallsiteinds_by_tag(partmps::PartitionedMPS; tag=tag)
    return findallsiteinds_by_tag(only.(siteinds(partmps)); tag=tag)
end
