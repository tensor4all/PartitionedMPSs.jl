using PartitionedMPSs
using Documenter

DocMeta.setdocmeta!(
    PartitionedMPSs, :DocTestSetup, :(using PartitionedMPSs); recursive=true
)

makedocs(;
    modules=[PartitionedMPSs],
    authors="Hiroshi Shinaoka <h.shinaoka@gmail.com> and contributors",
    sitename="PartitionedMPSs.jl",
    format=Documenter.HTML(;
        canonical="https://github.com/tensor4all/PartitionedMPSs.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=["Home" => "index.md"],
)

deploydocs(; repo="github.com/tensor4all/PartitionedMPSs.jl.git", devbranch="main")
