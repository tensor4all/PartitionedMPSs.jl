
function allequal(collection)
    if isempty(collection)
        return true
    end
    c = first(collection)
    return all([x == c for x in collection])
end

function Not(index::Int, length::Int)
    return vcat(1:(index - 1), (index + 1):length)
end

function typesafe_iterators_product(::Val{N}, dims) where {N}
    return Iterators.product(ntuple(i -> 1:dims[i], N)...)
end

_getindex(x, indices) = ntuple(i -> x[indices[i]], length(indices))

function _contract(
    a::AbstractArray{T1,N1},
    b::AbstractArray{T2,N2},
    idx_a::NTuple{n1,Int},
    idx_b::NTuple{n2,Int},
) where {T1,T2,N1,N2,n1,n2}
    length(idx_a) == length(idx_b) || error("length(idx_a) != length(idx_b)")
    # check if idx_a contains only unique elements
    length(unique(idx_a)) == length(idx_a) || error("idx_a contains duplicate elements")
    # check if idx_b contains only unique elements
    length(unique(idx_b)) == length(idx_b) || error("idx_b contains duplicate elements")
    # check if idx_a and idx_b are subsets of 1:N1 and 1:N2
    all(1 <= idx <= N1 for idx in idx_a) || error("idx_a contains elements out of range")
    all(1 <= idx <= N2 for idx in idx_b) || error("idx_b contains elements out of range")

    rest_idx_a = setdiff(1:N1, idx_a)
    rest_idx_b = setdiff(1:N2, idx_b)

    amat = reshape(
        permutedims(a, (rest_idx_a..., idx_a...)),
        prod(_getindex(size(a), rest_idx_a)),
        prod(_getindex(size(a), idx_a)),
    )
    bmat = reshape(
        permutedims(b, (idx_b..., rest_idx_b...)),
        prod(_getindex(size(b), idx_b)),
        prod(_getindex(size(b), rest_idx_b)),
    )

    return reshape(
        amat * bmat, _getindex(size(a), rest_idx_a)..., _getindex(size(b), rest_idx_b)...
    )
end

# QUESTION: Is this really a shallow copy? It works like a deep copy - Gianluca 
function shallowcopy(original)
    fieldnames = Base.fieldnames(typeof(original))
    new_fields = [Base.copy(getfield(original, name)) for name in fieldnames]
    return (typeof(original))(new_fields...)
end

function _asdiagonal(t, site::Index{T}; baseplev=0)::ITensor where {T<:Number}
    ITensors.hasinds(t, site') && error("Found $(site')")
    links = ITensors.uniqueinds(ITensors.inds(t), site)
    rawdata = Array(t, links..., site)
    tensor = zeros(eltype(t), size(rawdata)..., ITensors.dim(site))
    for i in 1:ITensors.dim(site)
        tensor[.., i, i] = rawdata[.., i]
    end
    return ITensor(
        tensor, links..., ITensors.prime(site, baseplev + 1), ITensors.prime(site, baseplev)
    )
end

function rearrange_siteinds(M::AbstractMPS, sites::Vector{Vector{Index{T}}})::MPS where {T}
    sitesold = siteinds(MPO(collect(M)))

    Set(Iterators.flatten(sites)) == Set(Iterators.flatten(sitesold)) ||
        error("siteinds do not match $(sites) != $(sitesold)")

    t = ITensor(1)
    tensors = Vector{ITensor}(undef, length(sites))
    tensors_old = collect(M)
    for (i, site) in enumerate(sites)
        for ind in site
            if ind ∈ inds(t)
                continue
            end
            contract_until = findfirst(x -> ind ∈ Set(collect(x)), inds.(tensors_old))
            contract_until !== nothing || error("ind $ind not found")
            for j in 1:contract_until
                t *= tensors_old[j]
            end
            for _ in 1:contract_until
                popfirst!(tensors_old)
            end
        end

        linds = if i > 1
            vcat(only(commoninds(t, tensors[i - 1])), sites[i])
        else
            sites[i]
        end
        tensors[i], t, _ = qr(t, linds)
    end
    tensors[end] *= t
    return MPS(tensors)
end

# A valid tag should not contain "=".
_valid_tag(tag::String)::Bool = !occursin("=", tag)

"""
Find sites with the given tag

For tag = `x`, if `sites` contains an Index object with `x`, the function returns a vector containing only its positon.

If not, the function seach for all Index objects with tags `x=1`, `x=2`, ..., and return their positions.

If no Index object is found, an empty vector will be returned.
"""
function findallsites_by_tag(
    sites::Vector{Index{T}}; tag::String="x", maxnsites::Int=1000
)::Vector{Int} where {T}
    _valid_tag(tag) || error("Invalid tag: $tag")

    # 1) Check if there is an Index with exactly `tag`
    if tag != ""
        idx = findall(hastags(tag), sites)
        if !isempty(idx)
            if length(idx) > 1
                error("Found more than one site index with tag $(tag)!")
            end
            return idx
        end
    end

    # 2) If not found, search for tag=1, tag=2, ...
    result = Int[]
    for n in 1:maxnsites
        tag_ = tag * "=$n"
        idx = findall(hastags(tag_), sites)
        if length(idx) == 0
            break
        elseif length(idx) > 1
            error("Found more than one site indices with $(tag_)!")
        end
        push!(result, idx[1])
    end
    return result
end

function findallsiteinds_by_tag(
    sites::AbstractVector{Index{T}}; tag::String="x", maxnsites::Int=1000
) where {T}
    _valid_tag(tag) || error("Invalid tag: $tag")
    positions = findallsites_by_tag(sites; tag=tag, maxnsites=maxnsites)
    return [sites[p] for p in positions]
end

function findallsites_by_tag(
    sites::Vector{Vector{Index{T}}}; tag::String="x", maxnsites::Int=1000
)::Vector{NTuple{2,Int}} where {T}
    _valid_tag(tag) || error("Invalid tag: $tag")

    sites_dict = Dict{Index{T},NTuple{2,Int}}()
    for i in 1:length(sites)
        for j in 1:length(sites[i])
            sites_dict[sites[i][j]] = (i, j)
        end
    end

    sitesflatten = collect(Iterators.flatten(sites))

    if tag != ""
        idx_exact = findall(i -> hastags(i, tag) && hasplev(i, 0), sitesflatten)
        if !isempty(idx_exact)
            if length(idx_exact) > 1
                error("Found more than one site index with tag '$tag'!")
            end
            # Return a single position
            return [sites_dict[sitesflatten[only(idx_exact)]]]
        end
    end

    result = NTuple{2,Int}[]
    for n in 1:maxnsites
        tag_ = tag * "=$n"
        idx = findall(i -> hastags(i, tag_) && hasplev(i, 0), sitesflatten)
        if length(idx) == 0
            break
        elseif length(idx) > 1
            error("Found more than one site indices with $(tag_)!")
        end

        push!(result, sites_dict[sitesflatten[only(idx)]])
    end
    return result
end

function findallsiteinds_by_tag(
    sites::Vector{Vector{Index{T}}}; tag::String="x", maxnsites::Int=1000
)::Vector{Index{T}} where {T}
    _valid_tag(tag) || error("Invalid tag: $tag")
    positions = findallsites_by_tag(sites; tag=tag, maxnsites=maxnsites)
    return [sites[i][j] for (i, j) in positions]
end

# FIXME: may be type unstable
# Gianluca: FIXED (?)
function _find_site_allplevs(
    tensor::ITensor, site::Index{T}; maxplev=10
)::Vector{Index{T}} where {T}
    ITensors.plev(site) == 0 || error("Site index must be unprimed.")
    return [
        ITensors.prime(site, plev) for
        plev in 0:maxplev if ITensors.prime(site, plev) ∈ ITensors.inds(tensor)
    ]
end

function makesitediagonal(M::AbstractMPS, tag::String)::MPS
    M_ = deepcopy(MPO(collect(M)))
    sites = siteinds(M_)

    target_positions = findallsites_by_tag(siteinds(M_); tag=tag)

    for t in eachindex(target_positions)
        i, j = target_positions[t]
        M_[i] = _asdiagonal(M_[i], sites[i][j])
    end

    return MPS(collect(M_))
end

function _extract_diagonal(t, site::Index{T}, site2::Index{T}) where {T<:Number}
    dim(site) == dim(site2) || error("Dimension mismatch")
    restinds = uniqueinds(inds(t), site, site2)
    newdata = zeros(eltype(t), dim.(restinds)..., dim(site))
    olddata = Array(t, restinds..., site, site2)
    for i in 1:dim(site)
        newdata[.., i] = olddata[.., i, i]
    end
    return ITensor(newdata, restinds..., site)
end

"""
Contract two adjacent tensors in MPO
"""
function combinesites(M::MPO, site1::Index, site2::Index)
    p1 = findsite(M, site1)
    p2 = findsite(M, site2)
    p1 === nothing && error("Not found $site1")
    p2 === nothing && error("Not found $site2")
    abs(p1 - p2) == 1 || error(
        "$site1 and $site2 are found at indices $p1 and $p2. They must be on two adjacent sites.",
    )
    tensors = ITensors.data(M)
    idx = min(p1, p2)
    tensor = tensors[idx] * tensors[idx + 1]
    deleteat!(tensors, idx:(idx + 1))
    insert!(tensors, idx, tensor)
    return MPO(tensors)
end

function combinesites(
    sites::Vector{Vector{Index{IndsT}}},
    site1::AbstractVector{Index{IndsT}},
    site2::AbstractVector{Index{IndsT}},
) where {IndsT}
    length(site1) == length(site2) || error("Length mismatch")
    for (s1, s2) in zip(site1, site2)
        sites = combinesites(sites, s1, s2)
    end
    return sites
end

function combinesites(
    sites::Vector{Vector{Index{IndsT}}}, site1::Index, site2::Index
) where {IndsT}
    sites = deepcopy(sites)
    p1 = findfirst(x -> x[1] == site1, sites)
    p2 = findfirst(x -> x[1] == site2, sites)
    if p1 === nothing || p2 === nothing
        error("Site not found")
    end
    if abs(p1 - p2) != 1
        error("Sites are not adjacent")
    end
    deleteat!(sites, min(p1, p2))
    deleteat!(sites, min(p1, p2))
    insert!(sites, min(p1, p2), [site1, site2])
    return sites
end
