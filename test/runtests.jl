import PartitionedMPSs as PM
using Random
using ITensors
using ITensorMPS
using Test

include("_util.jl")

include("projector_tests.jl")
include("subdomainmps_tests.jl")
include("partitionedmps_tests.jl")
include("contract_tests.jl")
include("patching_tests.jl")
include("util_tests.jl")
include("automul_tests.jl")

# include("bak/conversion_tests.jl")
