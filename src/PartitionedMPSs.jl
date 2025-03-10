module PartitionedMPSs

import OrderedCollections: OrderedSet, OrderedDict
using EllipsisNotation
using LinearAlgebra: LinearAlgebra

import ITensors: ITensors, Index, ITensor, dim, inds, qr, commoninds, uniqueinds, hasplev
import ITensorMPS: ITensorMPS, AbstractMPS, MPS, MPO, siteinds, findsites, findsite
import ITensors.TagSets: hastag, hastags

import FastMPOContractions as FMPOC

using Distributed
using Base.Threads

default_cutoff() = 1e-25
default_maxdim() = typemax(Int)

include("util.jl")
include("projector.jl")
include("subdomainmps.jl")
include("partitionedmps.jl")
include("patching.jl")
include("contract.jl")
include("adaptivemul.jl")
include("automul.jl")

end
