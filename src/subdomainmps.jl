"""
An MPS with a projector.
"""
struct SubDomainMPS
    data::MPS
    projector::Projector

    function SubDomainMPS(data::AbstractMPS, projector::Projector)
        _iscompatible(projector, data) || error(
            "Incompatible projector and data. Even small numerical noise can cause this error.",
        )
        projector = _trim_projector(data, projector)
        return new(MPS([x for x in data]), projector)
    end
end

siteinds(obj::SubDomainMPS) = collect(ITensors.siteinds(MPO([x for x in obj.data])))

_allsites(Ψ::AbstractMPS) = collect(Iterators.flatten(ITensors.siteinds(MPO(collect(Ψ)))))
_allsites(Ψ::SubDomainMPS) = _allsites(Ψ.data)

maxlinkdim(Ψ::SubDomainMPS) = ITensorMPS.maxlinkdim(Ψ.data)
maxbonddim(Ψ::SubDomainMPS) = ITensorMPS.maxlinkdim(Ψ.data)

function _trim_projector(obj::AbstractMPS, projector)
    sites = Set(_allsites(obj))
    newprj = deepcopy(projector)
    for (k, v) in newprj.data
        if !(k in sites)
            delete!(newprj.data, k)
        end
    end
    return newprj
end

function SubDomainMPS(Ψ::AbstractMPS)
    return SubDomainMPS(Ψ, Projector())
end

# Conversion Functions
ITensorMPS.MPS(projΨ::SubDomainMPS) = projΨ.data

function project(tensor::ITensor, projector::Projector)
    slice = Union{Int,Colon}[
        isprojectedat(projector, idx) ? projector[idx] : Colon() for
        idx in ITensors.inds(tensor)
    ]
    data_org = Array(tensor, ITensors.inds(tensor)...)
    data_trim = zero(data_org)
    if all(broadcast(x -> x isa Integer, slice))
        data_trim[slice...] = data_org[slice...]
    else
        data_trim[slice...] .= data_org[slice...]
    end
    return ITensor(data_trim, ITensors.inds(tensor)...)
end

function project(projΨ::SubDomainMPS, projector::Projector)::Union{Nothing,SubDomainMPS}
    newprj = projector & projΨ.projector
    if newprj === nothing
        return nothing
    end

    return SubDomainMPS(
        MPS([project(projΨ.data[n], newprj) for n in 1:length(projΨ.data)]), newprj
    )
end

function project(
    projΨ::SubDomainMPS, pairs::Vararg{Pair{Index{T},Int}}
)::Union{Nothing,SubDomainMPS} where {T}
    return project(projΨ, Projector(pairs...))
end

function project(
    Ψ::AbstractMPS, pairs::Vararg{Pair{Index{T},Int}}
)::Union{Nothing,SubDomainMPS} where {T}
    return project(Ψ, Projector(pairs...))
end

function project(Ψ::AbstractMPS, projector::Projector)::Union{Nothing,SubDomainMPS}
    return project(SubDomainMPS(Ψ), projector)
end

function project(
    projΨ::SubDomainMPS, projector::Dict{InsT,Int}
)::Union{Nothing,SubDomainMPS} where {InsT}
    return project(projΨ, Projector(projector))
end

function project(
    Ψ::AbstractMPS, projector::Dict{InsT,Int}
)::Union{Nothing,SubDomainMPS} where {InsT}
    return project(SubDomainMPS(Ψ), Projector(projector))
end

function _iscompatible(projector::Projector, tensor::ITensor)
    # Lazy implementation
    return ITensors.norm(project(tensor, projector) - tensor) == 0.0
end

function _iscompatible(projector::Projector, Ψ::AbstractMPS)
    return all((_iscompatible(projector, x) for x in Ψ))
end

function rearrange_siteinds(subdmps::SubDomainMPS, sites)
    mps_rearranged = rearrange_siteinds(MPS(subdmps), sites)
    return project(SubDomainMPS(mps_rearranged), subdmps.projector)
end

# Miscellaneous Functions
function Base.show(io::IO, obj::SubDomainMPS)
    return print(io, "SubDomainMPS projected on $(obj.projector.data)")
end

function prime(Ψ::SubDomainMPS, args...; kwargs...)
    return SubDomainMPS(
        ITensors.prime(MPS(Ψ), args...; kwargs...),
        ITensors.prime.(siteinds(Ψ), args...; kwargs...),
        Ψ.projector,
    )
end

function Base.isapprox(x::SubDomainMPS, y::SubDomainMPS; kwargs...)
    return Base.isapprox(x.data, y.data, kwargs...)
end

function isprojectedat(obj::SubDomainMPS, ind::IndsT)::Bool where {IndsT}
    return isprojectedat(obj.projector, ind)
end

function _fitsum(
    input_states::AbstractVector{T},
    init::T;
    coeffs::AbstractVector{<:Number}=ones(Int, length(input_states)),
    kwargs...,
) where {T}
    if !(:nsweeps ∈ keys(kwargs))
        kwargs = Dict{Symbol,Any}(kwargs)
        kwargs[:nsweeps] = 1
    end
    Ψs = [MPS(collect(x)) for x in input_states]
    init_Ψ = MPS(collect(init))
    res = FMPOC.fit(Ψs, init_Ψ; coeffs=coeffs, kwargs...)
    return T(collect(res))
end

function _add(ψ::AbstractMPS...; alg="fit", cutoff=1e-15, maxdim=typemax(Int), kwargs...)
    if alg == "directsum"
        return +(ITensors.Algorithm(alg), ψ...)
    elseif alg == "densitymatrix"
        if cutoff < 1e-15
            @warn "Cutoff is very small, it may suffer from numerical round errors. 
                    The densitymatrix algorithm squares the singular values of the reduce density matrix. 
                    Please consider increasing it or using fit algorithm."
        end
        return +(ITensors.Algorithm("densitymatrix"), ψ...; cutoff, maxdim, kwargs...)
    elseif alg == "fit"
        function f(x, y)
            return ITensors.truncate(
                +(ITensors.Algorithm("directsum"), x, y); cutoff, maxdim
            )
        end
        res_dm = reduce(f, ψ)
        res = _fitsum(collect(ψ), res_dm; cutoff, maxdim, kwargs...)
        return res
    else
        error("Unknown algorithm $(alg) for addition!")
    end
end

function Base.:+(
    Ψ::SubDomainMPS...; alg="directsum", cutoff=0.0, maxdim=typemax(Int), kwargs...
)::SubDomainMPS
    return _add(Ψ...; alg=alg, cutoff=cutoff, maxdim=maxdim, kwargs...)
end

function _add(
    Ψ::SubDomainMPS...; alg="directsum", cutoff=0.0, maxdim=typemax(Int), kwargs...
)::SubDomainMPS
    return project(
        _add([x.data for x in Ψ]...; alg=alg, cutoff=cutoff, maxdim=maxdim),
        reduce(|, [x.projector for x in Ψ]),
    )
end

function Base.:*(a::SubDomainMPS, b::Number)::SubDomainMPS
    return SubDomainMPS(a.data * b, a.projector)
end

function Base.:*(a::Number, b::SubDomainMPS)::SubDomainMPS
    return SubDomainMPS(b.data * a, b.projector)
end

function Base.:-(obj::SubDomainMPS)::SubDomainMPS
    return SubDomainMPS(-1 * obj.data, obj.projector)
end

function truncate(obj::SubDomainMPS; kwargs...)::SubDomainMPS
    return project(SubDomainMPS(ITensors.truncate(obj.data; kwargs...)), obj.projector)
end

function _norm(M::AbstractMPS)
    if ITensorMPS.isortho(M)
        return ITensors.norm(M[orthocenter(M)])
    end
    norm2_M = ITensors.dot(M, M)
    return sqrt(abs(norm2_M))
end

function LinearAlgebra.norm(M::SubDomainMPS)
    return _norm(MPS(M))
end

function _makesitediagonal(
    obj::SubDomainMPS, sites::AbstractVector{Index{IndsT}}; baseplev=0
) where {IndsT}
    M_ = deepcopy(MPO(collect(MPS(obj))))
    for site in sites
        target_site::Int = only(findsites(M_, site))
        M_[target_site] = _asdiagonal(M_[target_site], site; baseplev=baseplev)
    end

    newproj = deepcopy(obj.projector)
    for s in sites
        if isprojectedat(obj.projector, s)
            newproj.data[ITensors.prime(s, baseplev + 1)] = newproj.data[s]
            if baseplev != 0
                newproj.data[ITensors.prime(s, baseplev)] = newproj.data[s]
                delete!(newproj.data, s)
            end
        end
    end

    return project(M_, newproj)
end

function makesitediagonal(obj::SubDomainMPS, site::Index{IndsT}; baseplev=0) where {IndsT}
    return _makesitediagonal(obj, [site]; baseplev=baseplev)
end

function makesitediagonal(
    obj::SubDomainMPS, sites::AbstractVector{Index{IndsT}}; baseplev=0
) where {IndsT}
    return _makesitediagonal(obj, sites; baseplev=baseplev)
end

function makesitediagonal(obj::SubDomainMPS, tag::String)
    mps_diagonal = makesitediagonal(MPS(obj), tag)
    SubDomainMPS_diagonal = SubDomainMPS(mps_diagonal)

    target_sites = findallsiteinds_by_tag(
        unique(ITensors.noprime.(Iterators.flatten(siteinds(obj)))); tag=tag
    )

    newproj = deepcopy(obj.projector)
    for s in target_sites
        if isprojectedat(obj.projector, s)
            newproj[ITensors.prime(s)] = newproj[s]
        end
    end

    return project(SubDomainMPS_diagonal, newproj)
end

function extractdiagonal(
    obj::SubDomainMPS, sites::AbstractVector{Index{IndsT}}
) where {IndsT}
    tensors = collect(obj.data)
    for i in eachindex(tensors)
        for site in intersect(sites, ITensors.inds(tensors[i]))
            sitewithallplevs = _find_site_allplevs(tensors[i], site)
            tensors[i] = if length(sitewithallplevs) > 1
                tensors[i] = _extract_diagonal(tensors[i], sitewithallplevs...)
            else
                tensors[i]
            end
        end
    end

    newD = Dict{Index,Int}()
    # Duplicates of keys are discarded
    for (k, v) in obj.projector.data
        newk = ITensors.noprime(k)
        newD[newk] = v
    end
    return SubDomainMPS(MPS(tensors), Projector(newD))
end

function extractdiagonal(obj::SubDomainMPS, tag::String)::SubDomainMPS
    targetsites = findallsiteinds_by_tag(unique(ITensors.noprime.(_allsites(obj))); tag=tag)
    return extractdiagonal(obj, targetsites)
end

function extractdiagonal(subdmps::SubDomainMPS, site::Index{IndsT}) where {IndsT}
    return extractdiagonal(subdmps, [site])
end
