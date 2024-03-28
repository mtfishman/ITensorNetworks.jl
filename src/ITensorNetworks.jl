module ITensorNetworks

using AbstractTrees
using Combinatorics
using Compat
using DataGraphs
using DataStructures
using Dictionaries
using Distributions
using DocStringExtensions
using Graphs
using GraphsFlows
using Graphs.SimpleGraphs # AbstractSimpleGraph
using IsApprox
using ITensors
using ITensors.ContractionSequenceOptimization
using ITensors.ITensorVisualizationCore
using ITensors.LazyApply
using IterTools
using KrylovKit: KrylovKit
using LinearAlgebra
using NamedGraphs
using Observers
using Observers.DataFrames: select!
using PackageExtensionCompat
using Printf
using Requires
using SimpleTraits
using SparseArrayKit
using SplitApplyCombine
using StaticArrays
using Suppressor
using TimerOutputs
using StructWalk: StructWalk, WalkStyle, postwalk

using DataGraphs: IsUnderlyingGraph, edge_data_type, vertex_data_type
using Graphs: AbstractEdge, AbstractGraph, Graph, add_edge!
using ITensors:
  @Algorithm_str,
  @debug_check,
  @timeit_debug,
  δ,
  AbstractMPS,
  Algorithm,
  OneITensor,
  commontags,
  dim,
  orthocenter
using KrylovKit: exponentiate, eigsolve, linsolve
using NamedGraphs:
  AbstractNamedGraph,
  parent_graph,
  vertex_to_parent_vertex,
  parent_vertices_to_vertices,
  not_implemented

include("imports.jl")

# TODO: Move to `DataGraphs.jl`
edge_data_type(::AbstractNamedGraph) = Any
isassigned(::AbstractNamedGraph, ::Any) = false
function iterate(::AbstractDataGraph)
  return error(
    "Iterating data graphs is not yet defined. We may define it in the future as iterating through the vertex and edge data.",
  )
end

include("observers.jl")
include("visualize.jl")
include("graphs.jl")
include("itensors.jl")
include("abstractindsnetwork.jl")
include("indextags.jl")
include("indsnetwork.jl")
include("opsum.jl")
include("sitetype.jl")
include("abstractitensornetwork.jl")
include("contraction_sequences.jl")
include("models.jl")
include("tebd.jl")
include("itensornetwork.jl")
include("mincut.jl")
include("contract_deltas.jl")
include(joinpath("approx_itensornetwork", "utils.jl"))
include(joinpath("approx_itensornetwork", "density_matrix.jl"))
include(joinpath("approx_itensornetwork", "ttn_svd.jl"))
include(joinpath("approx_itensornetwork", "approx_itensornetwork.jl"))
include(joinpath("approx_itensornetwork", "partition.jl"))
include(joinpath("approx_itensornetwork", "binary_tree_partition.jl"))
include("contract.jl")
include("utility.jl")
include("specialitensornetworks.jl")
include("boundarymps.jl")
include("partitioneditensornetwork.jl")
include("edge_sequences.jl")
include(joinpath("formnetworks", "abstractformnetwork.jl"))
include(joinpath("formnetworks", "bilinearformnetwork.jl"))
include(joinpath("formnetworks", "quadraticformnetwork.jl"))
include(joinpath("caches", "beliefpropagationcache.jl"))
include("contraction_tree_to_graph.jl")
include("gauging.jl")
include("utils.jl")
include("tensornetworkoperators.jl")
include(joinpath("ITensorsExt", "itensorutils.jl"))
include(joinpath("Graphs", "abstractgraph.jl"))
include(joinpath("Graphs", "abstractdatagraph.jl"))
include(joinpath("solvers", "local_solvers", "eigsolve.jl"))
include(joinpath("solvers", "local_solvers", "exponentiate.jl"))
include(joinpath("solvers", "local_solvers", "dmrg_x.jl"))
include(joinpath("solvers", "local_solvers", "contract.jl"))
include(joinpath("solvers", "local_solvers", "linsolve.jl"))
include(joinpath("treetensornetworks", "abstracttreetensornetwork.jl"))
include(joinpath("treetensornetworks", "ttn.jl"))
include(joinpath("treetensornetworks", "opsum_to_ttn.jl"))
include(joinpath("treetensornetworks", "projttns", "abstractprojttn.jl"))
include(joinpath("treetensornetworks", "projttns", "projttn.jl"))
include(joinpath("treetensornetworks", "projttns", "projttnsum.jl"))
include(joinpath("treetensornetworks", "projttns", "projouterprodttn.jl"))
include(joinpath("solvers", "solver_utils.jl"))
include(joinpath("solvers", "defaults.jl"))
include(joinpath("solvers", "insert", "insert.jl"))
include(joinpath("solvers", "extract", "extract.jl"))
include(joinpath("solvers", "alternating_update", "alternating_update.jl"))
include(joinpath("solvers", "alternating_update", "region_update.jl"))
include(joinpath("solvers", "tdvp.jl"))
include(joinpath("solvers", "dmrg.jl"))
include(joinpath("solvers", "dmrg_x.jl"))
include(joinpath("solvers", "contract.jl"))
include(joinpath("solvers", "linsolve.jl"))
include(joinpath("solvers", "sweep_plans", "sweep_plans.jl"))
include("apply.jl")
include("inner.jl")
include("expect.jl")
include("environment.jl")

include("exports.jl")

function __init__()
  @require_extensions
  @require OMEinsumContractionOrders = "6f22d1fd-8eed-4bb7-9776-e7d684900715" include(
    joinpath("requires", "omeinsumcontractionorders.jl")
  )
end

end
