var documenterSearchIndex = {"docs":
[{"location":"","page":"Home","title":"Home","text":"CurrentModule = PartitionedMPSs","category":"page"},{"location":"#PartitionedMPSs","page":"Home","title":"PartitionedMPSs","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Documentation for PartitionedMPSs.","category":"page"},{"location":"","page":"Home","title":"Home","text":"","category":"page"},{"location":"","page":"Home","title":"Home","text":"Modules = [PartitionedMPSs]","category":"page"},{"location":"#PartitionedMPSs.LazyContraction","page":"Home","title":"PartitionedMPSs.LazyContraction","text":"Lazy evaluation for contraction of two SubDomainMPS objects.\n\n\n\n\n\n","category":"type"},{"location":"#PartitionedMPSs.PartitionedMPS","page":"Home","title":"PartitionedMPSs.PartitionedMPS","text":"PartitionedMPS is a structure that holds multiple MPSs (SubDomainMPS) that are associated with different non-overlapping projectors.\n\n\n\n\n\n","category":"type"},{"location":"#PartitionedMPSs.Projector","page":"Home","title":"PartitionedMPSs.Projector","text":"A projector represents a projection of a tensor from a set of its indices to integers. Each index is projected to a positive integer.\n\n\n\n\n\n","category":"type"},{"location":"#PartitionedMPSs.Projector-Union{Tuple{Pair{ITensors.Index{T}, Int64}}, Tuple{T}} where T","page":"Home","title":"PartitionedMPSs.Projector","text":"Constructing a projector from a single pair of index and integer.\n\n\n\n\n\n","category":"method"},{"location":"#PartitionedMPSs.SubDomainMPS","page":"Home","title":"PartitionedMPSs.SubDomainMPS","text":"An MPS with a projector.\n\n\n\n\n\n","category":"type"},{"location":"#Base.:&-Tuple{PartitionedMPSs.Projector, PartitionedMPSs.Projector}","page":"Home","title":"Base.:&","text":"a & b represents the intersection of the indices that a and b are projected at.\n\n\n\n\n\n","category":"method"},{"location":"#Base.:+-Tuple{PartitionedMPSs.PartitionedMPS, PartitionedMPSs.PartitionedMPS}","page":"Home","title":"Base.:+","text":"Add two PartitionedMPS objects.\n\nIf the two projects have the same projectors in the same order, the resulting PartitionedMPS will have the same projectors in the same order. By default, we use directsum algorithm to compute the sum and no truncation is performed.\n\n\n\n\n\n","category":"method"},{"location":"#Base.:<-Tuple{PartitionedMPSs.Projector, PartitionedMPSs.Projector}","page":"Home","title":"Base.:<","text":"a < b means that a is projected at a subset of the indices that b is projected at.\n\n\n\n\n\n","category":"method"},{"location":"#Base.:|-Tuple{PartitionedMPSs.Projector, PartitionedMPSs.Projector}","page":"Home","title":"Base.:|","text":"a | b represents the union of the indices that a and b are projected at.\n\nIf a is projected at inds=1 and b is not projected for the same inds, then a | b is not projected for inds.\n\n\n\n\n\n","category":"method"},{"location":"#Base.getindex-Tuple{PartitionedMPSs.PartitionedMPS, Integer}","page":"Home","title":"Base.getindex","text":"Indexing for PartitionedMPS. This is deprecated and will be removed in the future.\n\n\n\n\n\n","category":"method"},{"location":"#Base.isdisjoint-Tuple{AbstractVector{PartitionedMPSs.Projector}}","page":"Home","title":"Base.isdisjoint","text":"Return if projectors are not overlapping\n\n\n\n\n\n","category":"method"},{"location":"#Base.keys-Tuple{PartitionedMPSs.PartitionedMPS}","page":"Home","title":"Base.keys","text":"Return the keys, i.e., projectors of the PartitionedMPS.\n\n\n\n\n\n","category":"method"},{"location":"#Base.length-Tuple{PartitionedMPSs.PartitionedMPS}","page":"Home","title":"Base.length","text":"Get the number of the data in the PartitionedMPS. This is NOT the number of sites in the PartitionedMPS.\n\n\n\n\n\n","category":"method"},{"location":"#Base.values-Tuple{PartitionedMPSs.PartitionedMPS}","page":"Home","title":"Base.values","text":"Return the values, i.e., SubDomainMPS of the PartitionedMPS.\n\n\n\n\n\n","category":"method"},{"location":"#LinearAlgebra.norm-Tuple{PartitionedMPSs.PartitionedMPS}","page":"Home","title":"LinearAlgebra.norm","text":"Return the norm of the PartitionedMPS.\n\n\n\n\n\n","category":"method"},{"location":"#PartitionedMPSs._add_patching-Tuple{AbstractVector{PartitionedMPSs.SubDomainMPS}}","page":"Home","title":"PartitionedMPSs._add_patching","text":"Add multiple SubDomainMPS objects on the same projector.\n\nIf the bond dimension of the result reaches maxdim, perform patching recursively to reduce the bond dimension.\n\n\n\n\n\n","category":"method"},{"location":"#PartitionedMPSs._next_projindex-Tuple{PartitionedMPSs.Projector, Any}","page":"Home","title":"PartitionedMPSs._next_projindex","text":"Return the next index to be projected.\n\n\n\n\n\n","category":"method"},{"location":"#PartitionedMPSs.adaptive_patching-Tuple{PartitionedMPSs.PartitionedMPS, Any}","page":"Home","title":"PartitionedMPSs.adaptive_patching","text":"Adaptive patching\n\nDo patching recursively to reduce the bond dimension. If the bond dimension of a SubDomainMPS exceeds maxdim, perform patching.\n\n\n\n\n\n","category":"method"},{"location":"#PartitionedMPSs.adaptive_patching-Tuple{PartitionedMPSs.SubDomainMPS, Any}","page":"Home","title":"PartitionedMPSs.adaptive_patching","text":"Adaptive patching\n\nDo patching recursively to reduce the bond dimension. If the bond dimension of a SubDomainMPS exceeds maxdim, perform patching.\n\n\n\n\n\n","category":"method"},{"location":"#PartitionedMPSs.adaptivecontract","page":"Home","title":"PartitionedMPSs.adaptivecontract","text":"Perform contruction of two PartitionedMPS objects.\n\nThe SubDomainMPS objects of each PartitionedMPS do not overlap with each other. This makes the algorithm much simpler\n\n\n\n\n\n","category":"function"},{"location":"#PartitionedMPSs.add_patching-Tuple{AbstractVector{PartitionedMPSs.PartitionedMPS}}","page":"Home","title":"PartitionedMPSs.add_patching","text":"Add multiple PartitionedMPS objects.\n\n\n\n\n\n","category":"method"},{"location":"#PartitionedMPSs.contract!-Tuple{PartitionedMPSs.PartitionedMPS, PartitionedMPSs.PartitionedMPS, PartitionedMPSs.PartitionedMPS}","page":"Home","title":"PartitionedMPSs.contract!","text":"Contract two PartitionedMPS objects.\n\nExisting blocks M in the resulting PartitionedMPS will be overwritten if overwrite=true.\n\n\n\n\n\n","category":"method"},{"location":"#PartitionedMPSs.contract-Tuple{PartitionedMPSs.PartitionedMPS, PartitionedMPSs.PartitionedMPS}","page":"Home","title":"PartitionedMPSs.contract","text":"Contract two Blocked MPS objects.\n\nAt each site, the objects must share at least one site index.\n\n\n\n\n\n","category":"method"},{"location":"#PartitionedMPSs.extractdiagonal-Tuple{PartitionedMPSs.PartitionedMPS, Any}","page":"Home","title":"PartitionedMPSs.extractdiagonal","text":"Extract diagonal of the PartitionedMPS for s, s', ... for a given site index s, where s must have a prime level of 0.\n\n\n\n\n\n","category":"method"},{"location":"#PartitionedMPSs.makesitediagonal-Tuple{PartitionedMPSs.PartitionedMPS, Any}","page":"Home","title":"PartitionedMPSs.makesitediagonal","text":"Make the PartitionedMPS diagonal for a given site index s by introducing a dummy index s'.\n\n\n\n\n\n","category":"method"},{"location":"#PartitionedMPSs.projcontract-Tuple{AbstractVector{PartitionedMPSs.SubDomainMPS}, AbstractVector{PartitionedMPSs.SubDomainMPS}, PartitionedMPSs.Projector}","page":"Home","title":"PartitionedMPSs.projcontract","text":"Project two SubDomainMPS objects to proj before contracting them. The results are summed.\n\n\n\n\n\n","category":"method"},{"location":"#PartitionedMPSs.projcontract-Tuple{PartitionedMPSs.SubDomainMPS, PartitionedMPSs.SubDomainMPS, PartitionedMPSs.Projector}","page":"Home","title":"PartitionedMPSs.projcontract","text":"Project two SubDomainMPS objects to proj before contracting them.\n\n\n\n\n\n","category":"method"},{"location":"#PartitionedMPSs.project-Tuple{PartitionedMPSs.LazyContraction, PartitionedMPSs.Projector}","page":"Home","title":"PartitionedMPSs.project","text":"Project the LazyContraction object to prj before evaluating it.\n\nThis may result in projecting the external indices of a and b.\n\n\n\n\n\n","category":"method"},{"location":"#PartitionedMPSs.rearrange_siteinds-Tuple{PartitionedMPSs.PartitionedMPS, Any}","page":"Home","title":"PartitionedMPSs.rearrange_siteinds","text":"Rearrange the site indices of the PartitionedMPS according to the given order. If nessecary, tensors are fused or split to match the new order.\n\n\n\n\n\n","category":"method"},{"location":"#PartitionedMPSs.siteindices-Tuple{PartitionedMPSs.PartitionedMPS}","page":"Home","title":"PartitionedMPSs.siteindices","text":"Return the site indices of the PartitionedMPS. The site indices are returned as a vector of sets, where each set corresponds to the site indices at each site.\n\n\n\n\n\n","category":"method"},{"location":"#PartitionedMPSs.truncate-Tuple{PartitionedMPSs.PartitionedMPS}","page":"Home","title":"PartitionedMPSs.truncate","text":"Truncate a PartitionedMPS object piecewise.\n\nEach SubDomainMPS in the PartitionedMPS is truncated independently, but the cutoff is adjusted according to the norm of each SubDomainMPS. The total error is the sum of the errors in each SubDomainMPS.\n\n\n\n\n\n","category":"method"}]
}
