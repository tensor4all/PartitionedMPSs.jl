"""
A projector represents a projection of a tensor from a set of its indices to integers.
Each index is projected to a positive integer.
"""
struct Projector
    data::Dict{Index,Int} # values must be positive.

    function Projector(data::Dict{Index,Int})
        for (k, v) in data
            if v <= 0
                error("Value at $(k) must be positive.")
            end
        end
        return new(data)
    end
end

function Projector(data::Dict{Index{T},Int}) where {T}
    return Projector(Dict{Index,Int}(k => v for (k, v) in data))
end

function Projector(pairs::Vararg{Pair{Index{T},Int}}) where {T}
    return Projector(Dict{Index,Int}(pairs))
end

function Projector()
    return Projector(Dict{Index,Int}())
end

"""
Constructing a projector from a single pair of index and integer.
"""
function Projector(singleproj::Pair{Index{T},Int}) where {T}
    return Projector(Dict{Index,Int}(singleproj.first => singleproj.second))
end

function Base.hash(p::Projector, h::UInt)
    tmp = hash(
        sort(collect(Iterators.flatten(((hash(k, h), hash(v, h)) for (k, v) in p.data))))
    )
    return Base.hash(tmp, h)
end

projectedinds(p::Projector) = p.data

Base.getindex(p::Projector, inds) = p.data[inds]
Base.getindex(p::Projector, ind, default) = get(p.data, ind, default)
Base.keys(p::Projector) = keys(p.data)

function Base.iterate(obj::Projector, state)
    return iterate(obj.data, state)
end

function Base.iterate(obj::Projector)
    return iterate(obj.data)
end

Base.:(==)(a::Projector, b::Projector)::Bool = (a.data == b.data)

"""
`a < b` means that `a` is projected at a subset of the indices that `b` is projected at.
"""
function Base.:(<)(a::Projector, b::Projector)::Bool
    for k in intersect(keys(a.data), keys(b.data))
        if a[k] != b[k]
            return false
        end
    end

    return keys(b.data) ⊆ keys(a.data)
end

Base.:(>)(a::Projector, b::Projector)::Bool = (b < a)

"""
`a & b` represents the intersection of the indices that `a` and `b` are projected at.
"""
function Base.:&(a::Projector, b::Projector)::Union{Nothing,Projector}
    for k in intersect(keys(a.data), keys(b.data))
        if a[k] != b[k]
            return nothing
        end
    end
    return Projector(merge(a.data, b.data))
end

"""
`a | b` represents the union of the indices that `a` and `b` are projected at.

If `a` is projected at `inds=1` and `b` is not projected for the same `inds`, then `a | b` is not projected for `inds`.
"""
function Base.:|(a::Projector, b::Projector)::Projector
    data = Dict{Index,Int}()
    for k in intersect(keys(a.data), keys(b.data))
        if a[k] == b[k]
            data[k] = a[k]
        end
    end
    return Projector(data)
end

Base.:<=(a::Projector, b::Projector)::Bool = (a < b) || (a == b)
Base.:>=(a::Projector, b::Projector) = (b <= a)

function hasoverlap(p1, p2)::Bool
    return p1 & p2 !== nothing
end

function isprojectedat(p::Projector, ind::Index{T})::Bool where {T}
    return ind ∈ keys(p.data)
end

"""
Return if projectors are not overlapping
"""
function Base.isdisjoint(projectors::AbstractVector{Projector})::Bool
    for (n, a) in enumerate(projectors), (m, b) in enumerate(projectors)
        if n != m
            if hasoverlap(a, b)
                return false
            end
        end
    end
    return true
end

"""
Remove prime level in projected indices.
"""
function noprime(
    p::Projector; targetsites::Union{Nothing,AbstractVector{Index{T}}}=nothing
)::Projector where {T}
    if isnothing(targetsites)
        targetsites = keys(p)
    elseif targetsites ⊈ keys(p)
        error("Target sites are not projected indices.")
    end

    if isempty(targetsites)
        return p
    end

    new_dict = Dict(k ∈ targetsites ? ITensors.noprime(k) => v : k => v for (k, v) in p)
    return Projector(new_dict)
end

"""
Prime projected indices.
"""
function prime(p::Projector, plinc=1; kwargs...)::Projector
    targetsites = if :inds ∈ keys(kwargs)
        kwargs[:inds]
    else
        keys(p)
    end
    plev = if :plev ∈ keys(kwargs)
        kwargs[:plev]
    end
    isempty(targetsites) && return p

    function new_ind(k)
        if k ∉ targetsites
            return k
        end
        if isnothing(plev) || ITensors.hasplev(k, plev)
            return ITensors.prime(k, plinc)
        else
            return k
        end
    end

    new_dict = Dict(new_ind(k) => v for (k, v) in p)

    return Projector(new_dict)
end
