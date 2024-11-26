
"""
PartitionedMPS is a structure that holds multiple MPSs (SubDomainMPS) that are associated with different non-overlapping projectors.
"""
struct PartitionedMPS
    data::OrderedDict{Projector,SubDomainMPS}

    function PartitionedMPS(data::AbstractVector{SubDomainMPS})
        sites_all = [siteinds(prjmps) for prjmps in data]
        for n in 2:length(data)
            Set(sites_all[n]) == Set(sites_all[1]) || error("Sitedims mismatch")
        end
        isdisjoint([prjmps.projector for prjmps in data]) || error("Projectors are overlapping")

        dict_ = OrderedDict{Projector,SubDomainMPS}(
            data[i].projector => data[i] for i in 1:length(data)
        )
        return new(dict_)
    end
end

PartitionedMPS(data::SubDomainMPS) = PartitionedMPS([data])

"""
Return the site indices of the PartitionedMPS.
The site indices are returned as a vector of sets, where each set corresponds to the site indices at each site.
"""
function siteindices(obj::PartitionedMPS)
    return [Set(x) for x in ITensors.siteinds(first(values(obj.data)))]
end

ITensors.siteinds(obj::PartitionedMPS) = siteindices(obj)

"""
Get the number of sites in the PartitionedMPS
"""
Base.length(obj::PartitionedMPS) = length(first(obj.data))

"""
Indexing for PartitionedMPS. This is deprecated and will be removed in the future.
"""
function Base.getindex(bmps::PartitionedMPS, i::Integer)::SubDomainMPS
    @warn "Indexing for PartitionedMPS is deprecated. Use getindex(bmps, p::Projector) instead."
    return first(Iterators.drop(values(bmps.data), i - 1))
end

Base.getindex(obj::PartitionedMPS, p::Projector) = obj.data[p]

function Base.iterate(bmps::PartitionedMPS, state)
    return iterate(bmps.data, state)
end

function Base.iterate(bmps::PartitionedMPS)
    return iterate(bmps.data)
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
    return PartitionedMPS([rearrange_siteinds(prjmps, sites) for prjmps in values(obj)])
end

function prime(Ψ::PartitionedMPS, args...; kwargs...)
    return PartitionedMPS([prime(prjmps, args...; kwargs...) for prjmps in values(Ψ.data)])
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
    a::PartitionedMPS,
    b::PartitionedMPS;
    alg="directsum",
    cutoff=0.0,
    maxdim=typemax(Int),
    kwargs...,
)::PartitionedMPS
    data = SubDomainMPS[]
    for k in unique(vcat(collect(keys(a)), collect(keys(b)))) # preserve order
        if k ∈ keys(a) && k ∈ keys(b)
            a[k].projector == b[k].projector || error("Projectors mismatch at $(k)")
            push!(data, +(a[k], b[k]; alg, cutoff, maxdim, kwargs...))
        elseif k ∈ keys(a)
            push!(data, a[k])
        elseif k ∈ keys(b)
            push!(data, b[k])
        else
            error("Something went wrong")
        end
    end
    return PartitionedMPS(data)
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

function truncate(obj::PartitionedMPS; kwargs...)::PartitionedMPS
    return PartitionedMPS([truncate(v; kwargs...) for v in values(obj)])
end

# Only for debug
function ITensorMPS.MPS(obj::PartitionedMPS; cutoff=1e-25, maxdim=typemax(Int))::MPS
    return reduce(
        (x, y) -> truncate(+(x, y; alg="directsum"); cutoff, maxdim), values(obj.data)
    ).data # direct sum
end

function ITensorMPS.MPO(obj::PartitionedMPS; cutoff=1e-25, maxdim=typemax(Int))::MPO
    return MPO(collect(MPS(obj; cutoff=cutoff, maxdim=maxdim, kwargs...)))
end

"""
Make the PartitionedMPS diagonal for a given site index `s` by introducing a dummy index `s'`.
"""
function makesitediagonal(obj::PartitionedMPS, site)
    return PartitionedMPS([
        _makesitediagonal(prjmps, site; baseplev=baseplev) for prjmps in values(obj)
    ])
end

function _makesitediagonal(obj::PartitionedMPS, site; baseplev=0)
    return PartitionedMPS([
        _makesitediagonal(prjmps, site; baseplev=baseplev) for prjmps in values(obj)
    ])
end

"""
Extract diagonal of the PartitionedMPS for `s`, `s'`, ... for a given site index `s`,
where `s` must have a prime level of 0.
"""
function extractdiagonal(obj::PartitionedMPS, site)
    return PartitionedMPS([extractdiagonal(prjmps, site) for prjmps in values(obj)])
end
