using AbstractTrees: Leaves, PostOrderDFS
using Combinatorics: powerset

# a large number to prevent this edge being a cut
MAX_WEIGHT = 1e32

"""
Outputs a maximimally unbalanced directed binary tree DataGraph defining the desired graph structure
"""
function path_graph_structure(tn::ITensorNetwork)
  return path_graph_structure(tn, noncommoninds(Vector{ITensor}(tn)...))
end

"""
Given a `tn` and `outinds` (a subset of noncommoninds of `tn`), outputs a maximimally unbalanced
directed binary tree DataGraph of `outinds` defining the desired graph structure
"""
function path_graph_structure(tn::ITensorNetwork, outinds::Vector{<:Index})
  return _binary_tree_structure(tn, outinds; maximally_unbalanced=true)
end

"""
Outputs a directed binary tree DataGraph defining the desired graph structure
"""
function binary_tree_structure(tn::ITensorNetwork)
  return binary_tree_structure(tn, noncommoninds(Vector{ITensor}(tn)...))
end

"""
Given a `tn` and `outinds` (a subset of noncommoninds of `tn`), outputs a
directed binary tree DataGraph of `outinds` defining the desired graph structure
"""
function binary_tree_structure(tn::ITensorNetwork, outinds::Vector{<:Index})
  return _binary_tree_structure(tn, outinds; maximally_unbalanced=false)
end

"""
Calculate the mincut between two subsets of the uncontracted inds
(source_inds and terminal_inds) of the input tn.
Mincut of two inds list is defined as the mincut of two newly added vertices,
each one neighboring to one inds subset.
"""
function _mincut(
  tn::ITensorNetwork, source_inds::Vector{<:Index}, terminal_inds::Vector{<:Index}
)
  @assert length(source_inds) >= 1
  @assert length(terminal_inds) >= 1
  noncommon_inds = noncommoninds(Vector{ITensor}(tn)...)
  @assert issubset(source_inds, noncommon_inds)
  @assert issubset(terminal_inds, noncommon_inds)
  tn = disjoint_union(
    ITensorNetwork([ITensor(source_inds...), ITensor(terminal_inds...)]), tn
  )
  return GraphsFlows.mincut(tn, (1, 1), (2, 1), weights(tn))
end

"""
Calculate the mincut_partitions between two subsets of the uncontracted inds
(source_inds and terminal_inds) of the input tn.
"""
function _mincut_partitions(
  tn::ITensorNetwork, source_inds::Vector{<:Index}, terminal_inds::Vector{<:Index}
)
  p1, p2, cut = _mincut(tn, source_inds, terminal_inds)
  p1 = [v[1] for v in p1 if v[2] == 2]
  p2 = [v[1] for v in p2 if v[2] == 2]
  return p1, p2
end

function _mincut_partition_maxweightoutinds(
  tn::ITensorNetwork, source_inds::Vector{<:Index}, terminal_inds::Vector{<:Index}
)
  tn, out_to_maxweight_ind = _maxweightoutinds_tn(tn, [source_inds..., terminal_inds...])
  source_inds = [out_to_maxweight_ind[i] for i in source_inds]
  terminal_inds = [out_to_maxweight_ind[i] for i in terminal_inds]
  return _mincut_partitions(tn, source_inds, terminal_inds)
end

"""
Sum of shortest path distances among all outinds.
"""
function _distance(tn::ITensorNetwork, outinds::Vector{<:Index})
  @assert length(outinds) >= 1
  @assert issubset(outinds, noncommoninds(Vector{ITensor}(tn)...))
  if length(outinds) == 1
    return 0.0
  end
  new_tensors = [ITensor(i) for i in outinds]
  tn = disjoint_union(ITensorNetwork(new_tensors), tn)
  distances = 0.0
  for i in 1:(length(new_tensors) - 1)
    ds = dijkstra_shortest_paths(tn, [(i, 1)], weights(tn))
    for j in (i + 1):length(new_tensors)
      distances += ds.dists[(j, 1)]
    end
  end
  return distances
end

"""
create a tn with empty ITensors whose outinds weights are MAX_WEIGHT
The maxweight_tn is constructed so that only commoninds of the tn
will be considered in mincut.
"""
function _maxweightoutinds_tn(tn::ITensorNetwork, outinds::Union{Nothing,Vector{<:Index}})
  @assert issubset(outinds, noncommoninds(Vector{ITensor}(tn)...))
  out_to_maxweight_ind = Dict{Index,Index}()
  for ind in outinds
    out_to_maxweight_ind[ind] = Index(MAX_WEIGHT, ind.tags)
  end
  maxweight_tn = copy(tn)
  for v in vertices(maxweight_tn)
    t = maxweight_tn[v]
    inds1 = [i for i in inds(t) if !(i in outinds)]
    inds2 = [out_to_maxweight_ind[i] for i in inds(t) if i in outinds]
    newt = ITensor(inds1..., inds2...)
    maxweight_tn[v] = newt
  end
  return maxweight_tn, out_to_maxweight_ind
end

"""
Given a tn and outinds (a subset of noncommoninds of tn), get a `DataGraph`
with binary tree structure of outinds that will be used in the binary tree partition.
If maximally_unbalanced=true, the binary tree will have a line/mps structure.
The binary tree is recursively constructed from leaves to the root.

Example:
# TODO
"""
function _binary_tree_structure(
  tn::ITensorNetwork, outinds::Vector{<:Index}; maximally_unbalanced::Bool=false
)
  inds_tree_vector = _binary_tree_partition_inds(
    tn, outinds; maximally_unbalanced=maximally_unbalanced
  )
  return _nested_vector_to_directed_tree(inds_tree_vector)
end

function _binary_tree_partition_inds(
  tn::ITensorNetwork, outinds::Vector{<:Index}; maximally_unbalanced::Bool=false
)
  if length(outinds) == 1
    return outinds
  end
  maxweight_tn, out_to_maxweight_ind = _maxweightoutinds_tn(tn, outinds)
  tn_pair = tn => maxweight_tn
  if maximally_unbalanced == false
    return _binary_tree_partition_inds_mincut(tn_pair, out_to_maxweight_ind)
  else
    return line_to_tree(
      _binary_tree_partition_inds_maximally_unbalanced(tn_pair, out_to_maxweight_ind)
    )
  end
end

function _nested_vector_to_directed_tree(inds_tree_vector::Vector)
  if length(inds_tree_vector) == 1 && inds_tree_vector[1] isa Index
    inds_btree = DataGraph(NamedDiGraph([1]), Index)
    inds_btree[1] = inds_tree_vector[1]
    return inds_btree
  end
  treenode_to_v = Dict{Union{Vector,Index},Int}()
  graph = DataGraph(NamedDiGraph(), Index)
  v = 1
  for treenode in PostOrderDFS(inds_tree_vector)
    add_vertex!(graph, v)
    treenode_to_v[treenode] = v
    if treenode isa Index
      graph[v] = treenode
    else
      @assert length(treenode) == 2
      add_edge!(graph, v, treenode_to_v[treenode[1]])
      add_edge!(graph, v, treenode_to_v[treenode[2]])
    end
    v += 1
  end
  return graph
end

"""
Given a tn and outinds, returns a vector of indices representing MPS inds ordering.
"""
function _mps_partition_inds_order(
  tn::ITensorNetwork, outinds::Union{Nothing,Vector{<:Index}}
)
  if outinds == nothing
    outinds = noncommoninds(Vector{ITensor}(tn)...)
  end
  if length(outinds) == 1
    return outinds
  end
  tn2, out_to_maxweight_ind = _maxweightoutinds_tn(tn, outinds)
  return _binary_tree_partition_inds_maximally_unbalanced(tn => tn2, out_to_maxweight_ind)
end

function _binary_tree_partition_inds_maximally_unbalanced(
  tn_pair::Pair{<:ITensorNetwork,<:ITensorNetwork}, out_to_maxweight_ind::Dict{Index,Index}
)
  outinds = collect(keys(out_to_maxweight_ind))
  @assert length(outinds) >= 1
  if length(outinds) <= 2
    return outinds
  end
  first_inds, _ = _mincut_inds(
    tn_pair, out_to_maxweight_ind, collect(powerset(outinds, 1, 1))
  )
  first_ind = first_inds[1]
  linear_order = [first_ind]
  outinds = setdiff(outinds, linear_order)
  while length(outinds) > 1
    sourceinds_list = [Vector{Index}([linear_order..., i]) for i in outinds]
    target_inds, _ = _mincut_inds(tn_pair, out_to_maxweight_ind, sourceinds_list)
    new_ind = setdiff(target_inds, linear_order)[1]
    push!(linear_order, new_ind)
    outinds = setdiff(outinds, [new_ind])
  end
  push!(linear_order, outinds[1])
  return linear_order
end

function _binary_tree_partition_inds_mincut(
  tn_pair::Pair{<:ITensorNetwork,<:ITensorNetwork}, out_to_maxweight_ind::Dict{Index,Index}
)
  outinds = collect(keys(out_to_maxweight_ind))
  @assert length(outinds) >= 1
  if length(outinds) <= 2
    return outinds
  end
  while length(outinds) > 2
    tree_list = collect(powerset(outinds, 2, 2))
    sourceinds_list = [collect(Leaves(i)) for i in tree_list]
    _, i = _mincut_inds(tn_pair, out_to_maxweight_ind, sourceinds_list)
    tree = tree_list[i]
    outinds = setdiff(outinds, tree)
    outinds = vcat([tree], outinds)
  end
  return outinds
end

"""
Find a vector of indices within sourceinds_list yielding the mincut of given tn_pair.
Args:
  tn_pair: a pair of tns (tn1 => tn2), where tn2 is generated via _maxweightoutinds_tn(tn1)
  out_to_maxweight_ind: a dict mapping each out ind in tn1 to out ind in tn2
  sourceinds_list: a list of vector of indices to be considered
Note:
  For each sourceinds in sourceinds_list, we consider its mincut within both tns (tn1, tn2) given in tn_pair.
  The mincut in tn1 represents the rank upper bound when splitting sourceinds with other inds in outinds.
  The mincut in tn2 represents the rank upper bound when the weights of outinds are very large.
  The first mincut upper_bounds the number of non-zero singular values, while the second empirically reveals the
  singular value decay.
  We output the sourceinds where the first mincut value is the minimum, the secound mincut value is also
  the minimum under the condition that the first mincut is optimal, and the sourceinds have the lowest all-pair shortest path.
"""
function _mincut_inds(
  tn_pair::Pair{<:ITensorNetwork,<:ITensorNetwork},
  out_to_maxweight_ind::Dict{Index,Index},
  sourceinds_list::Vector{<:Vector{<:Index}},
)
  function _mincut_value(tn, sinds, outinds)
    tinds = setdiff(outinds, sinds)
    _, _, cut = _mincut(tn, sinds, tinds)
    return cut
  end
  function _get_weights(source_inds, outinds, maxweight_source_inds, maxweight_outinds)
    mincut_val = _mincut_value(tn_pair.first, source_inds, outinds)
    maxweight_mincut_val = _mincut_value(
      tn_pair.second, maxweight_source_inds, maxweight_outinds
    )
    dist = _distance(tn_pair.first, source_inds)
    return (mincut_val, maxweight_mincut_val, dist)
  end

  outinds = collect(keys(out_to_maxweight_ind))
  maxweight_outinds = collect(values(out_to_maxweight_ind))
  weights = []
  for source_inds in sourceinds_list
    maxweight_source_inds = [out_to_maxweight_ind[i] for i in source_inds]
    push!(
      weights, _get_weights(source_inds, outinds, maxweight_source_inds, maxweight_outinds)
    )
  end
  i = argmin(weights)
  return sourceinds_list[i], i
end
