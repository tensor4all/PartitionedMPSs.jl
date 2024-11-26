"""
By default, elementwise multiplication will be performed.

This function is kind of deprecated and will be removed in the future.
"""
function automul(
    M1::PartitionedMPS,
    M2::PartitionedMPS;
    tag_row::String="",
    tag_shared::String="",
    tag_col::String="",
    alg="naive",
    maxdim=typemax(Int),
    cutoff=1e-25,
    kwargs...,
)
    all(length.(siteinds(M1)) .== 1) || error("M1 should have only 1 site index per site")
    all(length.(siteinds(M2)) .== 1) || error("M2 should have only 1 site index per site")

    sites_row = _findallsiteinds_by_tag(M1; tag=tag_row)
    sites_shared = _findallsiteinds_by_tag(M1; tag=tag_shared)
    sites_col = _findallsiteinds_by_tag(M2; tag=tag_col)
    sites_matmul = Set(Iterators.flatten([sites_row, sites_shared, sites_col]))

    sites1 = only.(siteinds(M1))
    sites1_ewmul = setdiff(only.(siteinds(M1)), sites_matmul)
    sites2_ewmul = setdiff(only.(siteinds(M2)), sites_matmul)
    sites2_ewmul == sites1_ewmul || error("Invalid sites for elementwise multiplication")

    M1 = _makesitediagonal(M1, sites1_ewmul; baseplev=1)
    M2 = _makesitediagonal(M2, sites2_ewmul; baseplev=0)

    sites_M1_diag = [collect(x) for x in siteinds(M1)]
    sites_M2_diag = [collect(x) for x in siteinds(M2)]

    M1 = rearrange_siteinds(M1, combinesites(sites_M1_diag, sites_row, sites_shared))

    M2 = rearrange_siteinds(M2, combinesites(sites_M2_diag, sites_shared, sites_col))

    M = contract(M1, M2; alg=alg, kwargs...)

    M = extractdiagonal(M, sites1_ewmul)

    ressites = Vector{eltype(siteinds(M1)[1])}[]
    for s in siteinds(M)
        s_ = unique(ITensors.noprime.(s))
        if length(s_) == 1
            push!(ressites, s_)
        else
            if s_[1] ∈ sites1
                push!(ressites, [s_[1]])
                push!(ressites, [s_[2]])
            else
                push!(ressites, [s_[2]])
                push!(ressites, [s_[1]])
            end
        end
    end
    return truncate(rearrange_siteinds(M, ressites); cutoff=cutoff, maxdim=maxdim)
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

function _findallsiteinds_by_tag(M::PartitionedMPS; tag=tag)
    return findallsiteinds_by_tag(only.(siteinds(M)); tag=tag)
end

# The following code is copied from Quantics.jl

function findallsiteinds_by_tag(
    sites::AbstractVector{Index{T}}; tag::String="x", maxnsites::Int=1000
) where {T}
    _valid_tag(tag) || error("Invalid tag: $tag")
    positions = findallsites_by_tag(sites; tag=tag, maxnsites=maxnsites)
    return [sites[p] for p in positions]
end

function findallsites_by_tag(
    sites::Vector{Index{T}}; tag::String="x", maxnsites::Int=1000
)::Vector{Int} where {T}
    _valid_tag(tag) || error("Invalid tag: $tag")
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

_valid_tag(tag::String)::Bool = !occursin("=", tag)
