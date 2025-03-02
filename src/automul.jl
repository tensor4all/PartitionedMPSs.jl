@doc raw"""
    function elemmul(
        M1::PartitionedMPS,
        M2::PartitionedMPS
    )

Performs elementwise multiplication between partitioned MPSs. Element-wise product is defined 
as: 

```math
   (fg) (\xi) = f(\xi)g(\xi) = \sum_{\xi'} f(\xi, \xi') g(\xi')
```

where ``f(\xi, \xi') = f(\xi) \delta_{\xi, \xi'}``. 
"""
function elemmul(
    M1::PartitionedMPS,
    M2::PartitionedMPS;
    alg="zipup",
    maxdim=typemax(Int),
    cutoff=1e-25,
    kwargs...,
)
    all(length.(siteinds(M1)) .== 1) || error("M1 should have only 1 site index per site")
    all(length.(siteinds(M2)) .== 1) || error("M2 should have only 1 site index per site")

    only.(siteinds(M1)) == only.(siteinds(M2)) ||
        error("Sites for element wise multiplication should be identical")
    sites_ewmul = only.(siteinds(M1))

    M1 = makesitediagonal(M1, sites_ewmul; baseplev=1)
    M2 = makesitediagonal(M2, sites_ewmul; baseplev=0)

    M = contract(M1, M2; alg=alg, kwargs...)

    M = extractdiagonal(M, sites_ewmul)

    return truncate(M; cutoff=cutoff, maxdim=maxdim)
end

@doc raw"""
    function automul(
        M1::PartitionedMPS,
        M2::PartitionedMPS;
        tag_row::String="",
        tag_shared::String="",
        tag_col::String="",
        ...
)

Performs automatic multiplication between partitioned MPSs. Automatic multiplication is defined
as:

```math
   (fg) (\sigma_{row}, \sigma_{col}; \xi) = \sum_{\sigma_{shared}} 
   f(\sigma_{row}, \sigma_{shared}; \xi) g(\sigma_{shared}, \sigma_{col} ; \xi).
```

By default, only element-wise product on sites ``\xi`` will be performed. See also: [`elemmul`](@ref).
"""
function automul(
    M1::PartitionedMPS,
    M2::PartitionedMPS;
    tag_row::String="",
    tag_shared::String="",
    tag_col::String="",
    alg="zipup",
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

    M1 = makesitediagonal(M1, sites1_ewmul; baseplev=1)
    M2 = makesitediagonal(M2, sites2_ewmul; baseplev=0)

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
