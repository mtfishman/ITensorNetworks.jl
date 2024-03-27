module ITensorNetworks

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
  AbstractNamedGraph, parent_graph, parent_vertices_to_vertices, not_implemented

# TODO: Delete this.
include("imports.jl")

include("Graphs/abstractgraph.jl")
include("Graphs/abstractdatagraph.jl")
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
include("expect.jl")
include("models.jl")
include("tebd.jl")
include("itensornetwork.jl")
include("mincut.jl")
include("contract_deltas.jl")
include("approx_itensornetwork/utils.jl")
include("approx_itensornetwork/density_matrix.jl")
include("approx_itensornetwork/ttn_svd.jl")
include("approx_itensornetwork/approx_itensornetwork.jl")
include("approx_itensornetwork/partition.jl")
include("approx_itensornetwork/binary_tree_partition.jl")
include("contract.jl")
include("utility.jl")
include("specialitensornetworks.jl")
include("boundarymps.jl")
include("partitioneditensornetwork.jl")
include("edge_sequences.jl")
include("formnetworks/abstractformnetwork.jl")
include("formnetworks/bilinearformnetwork.jl")
include("formnetworks/quadraticformnetwork.jl")
include("caches/beliefpropagationcache.jl")
include("contraction_tree_to_graph.jl")
include("gauging.jl")
include("utils.jl")
include("tensornetworkoperators.jl")
include("ITensorsExt/itensorutils.jl")
include("solvers/local_solvers/eigsolve.jl")
include("solvers/local_solvers/exponentiate.jl")
include("solvers/local_solvers/dmrg_x.jl")
include("solvers/local_solvers/contract.jl")
include("solvers/local_solvers/linsolve.jl")
include("treetensornetworks/abstracttreetensornetwork.jl")
include("treetensornetworks/ttn.jl")
include("treetensornetworks/opsum_to_ttn.jl")
include("treetensornetworks/projttns/abstractprojttn.jl")
include("treetensornetworks/projttns/projttn.jl")
include("treetensornetworks/projttns/projttnsum.jl")
include("treetensornetworks/projttns/projouterprodttn.jl")
include("solvers/solver_utils.jl")
include("solvers/defaults.jl")
include("solvers/insert/insert.jl")
include("solvers/extract/extract.jl")
include("solvers/alternating_update/alternating_update.jl")
include("solvers/alternating_update/region_update.jl")
include("solvers/tdvp.jl")
include("solvers/dmrg.jl")
include("solvers/dmrg_x.jl")
include("solvers/contract.jl")
include("solvers/linsolve.jl")
include("solvers/sweep_plans/sweep_plans.jl")
include("apply.jl")
include("environment.jl")

include("exports.jl")

function __init__()
  @require_extensions
  @require OMEinsumContractionOrders = "6f22d1fd-8eed-4bb7-9776-e7d684900715" include(
    "requires/omeinsumcontractionorders.jl"
  )
end

end
