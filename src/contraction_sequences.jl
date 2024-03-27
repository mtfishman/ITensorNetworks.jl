using Graphs: vertices
using ITensors: ITensor, contract
using ITensors.ContractionSequenceOptimization: deepmap, optimal_contraction_sequence
using ITensors.NDTensors: Algorithm, @Algorithm_str
using NamedGraphs: Key

function contraction_sequence(tn::Vector{ITensor}; alg="optimal", kwargs...)
  return contraction_sequence(Algorithm(alg), tn; kwargs...)
end

function contraction_sequence(tn::AbstractITensorNetwork; kwargs...)
  seq_linear_index = contraction_sequence(Vector{ITensor}(tn); kwargs...)
  # TODO: Use Functors.fmap?
  return deepmap(n -> Key(vertices(tn)[n]), seq_linear_index)
end

function contraction_sequence(::Algorithm"optimal", tn::Vector{ITensor})
  return optimal_contraction_sequence(tn)
end

function contraction_sequence_requires_error(module_name, algorithm)
  return "Module `$(module_name)` not found, please type `using $(module_name)` before using the \"$(algorithm)\" contraction sequence backend!"
end

"""
    GreedyMethod(; method=MinSpaceOut(), nrepeat=10)

The fast but poor greedy optimizer. Input arguments are:

* `method` is `MinSpaceDiff()` or `MinSpaceOut`.
    * `MinSpaceOut` choose one of the contraction that produces a minimum output tensor size,
    * `MinSpaceDiff` choose one of the contraction that decrease the space most.
* `nrepeat` is the number of repeatition, returns the best contraction order.
"""
function contraction_sequence(::Algorithm"greedy", tn::Vector{ITensor}; kwargs...)
  if !isdefined(@__MODULE__, :OMEinsumContractionOrders)
    error(contraction_sequence_requires_error("OMEinsumContractionOrders", "greedy"))
  end
  return optimize_contraction_sequence(
    tn; optimizer=OMEinsumContractionOrders.GreedyMethod(; kwargs...)
  )
end

"""
    TreeSA(; sc_target=20, βs=collect(0.01:0.05:15), ntrials=10, niters=50,
             sc_weight=1.0, rw_weight=0.2, initializer=:greedy, greedy_config=GreedyMethod(; nrepeat=1))

Optimize the einsum contraction pattern using the simulated annealing on tensor expression tree.

* `sc_target` is the target space complexity,
* `ntrials`, `βs` and `niters` are annealing parameters, doing `ntrials` indepedent annealings, each has inverse tempteratures specified by `βs`, in each temperature, do `niters` updates of the tree.
* `sc_weight` is the relative importance factor of space complexity in the loss compared with the time complexity.
* `rw_weight` is the relative importance factor of memory read and write in the loss compared with the time complexity.
* `initializer` specifies how to determine the initial configuration, it can be `:greedy` or `:random`. If it is using `:greedy` method to generate the initial configuration, it also uses two extra arguments `greedy_method` and `greedy_nrepeat`.
* `nslices` is the number of sliced legs, default is 0.
* `fixed_slices` is a vector of sliced legs, default is `[]`.

### References
* [Recursive Multi-Tensor Contraction for XEB Verification of Quantum Circuits](https://arxiv.org/abs/2108.05665)
"""
function contraction_sequence(::Algorithm"tree_sa", tn; kwargs...)
  if !isdefined(@__MODULE__, :OMEinsumContractionOrders)
    error(contraction_sequence_requires_error("OMEinsumContractionOrders", "tree_sa"))
  end
  return optimize_contraction_sequence(
    tn; optimizer=OMEinsumContractionOrders.TreeSA(; kwargs...)
  )
end

"""
    SABipartite(; sc_target=25, ntrials=50, βs=0.1:0.2:15.0, niters=1000
                  max_group_size=40, greedy_config=GreedyMethod(), initializer=:random)

Optimize the einsum code contraction order using the Simulated Annealing bipartition + Greedy approach.
This program first recursively cuts the tensors into several groups using simulated annealing,
with maximum group size specifed by `max_group_size` and maximum space complexity specified by `sc_target`,
Then finds the contraction order inside each group with the greedy search algorithm. Other arguments are:

* `size_dict`, a dictionary that specifies leg dimensions,
* `sc_target` is the target space complexity, defined as `log2(number of elements in the largest tensor)`,
* `max_group_size` is the maximum size that allowed to used greedy search,
* `βs` is a list of inverse temperature `1/T`,
* `niters` is the number of iteration in each temperature,
* `ntrials` is the number of repetition (with different random seeds),
* `greedy_config` configures the greedy method,
* `initializer`, the partition configuration initializer, one can choose `:random` or `:greedy` (slow but better).

### References
* [Hyper-optimized tensor network contraction](https://arxiv.org/abs/2002.01935)
"""
function contraction_sequence(::Algorithm"sa_bipartite", tn; kwargs...)
  if !isdefined(@__MODULE__, :OMEinsumContractionOrders)
    error(contraction_sequence_requires_error("OMEinsumContractionOrders", "sa_bipartite"))
  end
  return optimize_contraction_sequence(
    tn; optimizer=OMEinsumContractionOrders.SABipartite(; kwargs...)
  )
end

"""
    KaHyParBipartite(; sc_target, imbalances=collect(0.0:0.005:0.8),
                       max_group_size=40, greedy_config=GreedyMethod())

Optimize the einsum code contraction order using the KaHyPar + Greedy approach.
This program first recursively cuts the tensors into several groups using KaHyPar,
with maximum group size specifed by `max_group_size` and maximum space complexity specified by `sc_target`,
Then finds the contraction order inside each group with the greedy search algorithm. Other arguments are:

* `sc_target` is the target space complexity, defined as `log2(number of elements in the largest tensor)`,
* `imbalances` is a KaHyPar parameter that controls the group sizes in hierarchical bipartition,
* `max_group_size` is the maximum size that allowed to used greedy search,
* `greedy_config` is a greedy optimizer.

### References
* [Hyper-optimized tensor network contraction](https://arxiv.org/abs/2002.01935)
* [Simulating the Sycamore quantum supremacy circuits](https://arxiv.org/abs/2103.03074)
"""
function contraction_sequence(::Algorithm"kahypar_bipartite", tn; kwargs...)
  if !isdefined(@__MODULE__, :OMEinsumContractionOrders)
    error(
      contraction_sequence_requires_error("OMEinsumContractionOrders", "kahypar_bipartite")
    )
  end
  return optimize_contraction_sequence(
    tn; optimizer=OMEinsumContractionOrders.KaHyParBipartite(; kwargs...)
  )
end
