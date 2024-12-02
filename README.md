# PartitionedMPSs

[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://tensor4all.github.io/PartitionedMPSs.jl/dev)
[![CI](https://github.com/tensor4all/PartitionedMPSs.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/tensor4all/PartitionedMPSs.jl/actions/workflows/CI.yml)

## General Information
This library provides an implementation of partitioned matrix product states (MPSs) in Julia. A `PartitionedMPS` (short for PartitionedMatrixProductStates) object is a collection of `SubDomainMPS` (short for SubDomain Matrix Product State) objects, each of which is an MPS projected on a subdomain of the whole index set.
This library also provides a block-wise multiplication of partitioned MPSs.


## Usage

Please refer to the [documentation](https://tensor4all.github.io/PartitionedMPSs.jl/) for usage.