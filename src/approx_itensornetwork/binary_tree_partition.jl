using DataGraphs: DataGraph
using ITensors: Index, ITensor, delta, noncommoninds, replaceinds, sim
using ITensors.NDTensors: Algorithm, @Algorithm_str
using NamedGraphs: disjoint_union, rename_vertices, subgraph

function _binary_partition(tn::ITensorNetwork, source_inds::Vector{<:Index})
  external_inds = noncommoninds(Vector{ITensor}(tn)...)
  # add delta tensor to each external ind
  external_sim_ind = [sim(ind) for ind in external_inds]
  tn = map_data(t -> replaceinds(t, external_inds => external_sim_ind), tn; edges=[])
  tn_wo_deltas = rename_vertices(v -> v[1], subgraph(v -> v[2] == 1, tn))
  deltas = Vector{ITensor}(subgraph(v -> v[2] == 2, tn))
  scalars = rename_vertices(v -> v[1], subgraph(v -> v[2] == 3, tn))
  new_deltas = [
    delta(external_inds[i], external_sim_ind[i]) for i in 1:length(external_inds)
  ]
  deltas = [deltas..., new_deltas...]
  tn = disjoint_union(tn_wo_deltas, ITensorNetwork(deltas), scalars)
  p1, p2 = _mincut_partition_maxweightoutinds(
    tn, source_inds, setdiff(external_inds, source_inds)
  )
  source_tn = _contract_deltas(subgraph(tn, p1))
  remain_tn = _contract_deltas(subgraph(tn, p2))
  outinds_source = noncommoninds(Vector{ITensor}(source_tn)...)
  outinds_remain = noncommoninds(Vector{ITensor}(remain_tn)...)
  common_inds = intersect(outinds_source, outinds_remain)
  @assert (
    length(external_inds) ==
    length(union(outinds_source, outinds_remain)) - length(common_inds)
  )
  # We want the output two tns be connected to each other, so that the output
  # of `binary_tree_partition` is a partition with a binary tree structure.
  # Below we check if `source_tn` and `remain_tn` are connected, if not adding
  # each tn a scalar tensor to force them to be connected.
  if common_inds == []
    @info "_binary_partition outputs are not connected"
    ind = Index(1, "unit_scalar_ind")
    t1 = ITensor([1.0], ind)
    t2 = ITensor([1.0], ind)
    v1 = (nv(scalars) + 1, 3)
    v2 = (nv(scalars) + 2, 3)
    add_vertex!(source_tn, v1)
    add_vertex!(remain_tn, v2)
    source_tn[v1] = t1
    remain_tn[v2] = t2
  end
  return source_tn, remain_tn
end

"""
Given an input tn and a rooted binary tree of indices, return a partition of tn with the
same binary tree structure as inds_btree.
Note: in the output partition, we add multiple delta tensors to the network so that
  the output graph is guaranteed to be the same binary tree as inds_btree.
Note: in the output partition, we add multiple scalar tensors. These tensors are used to
  make the output partition connected, even if the input `tn` is disconnected.
Note: in the output partition, tensor vertex names will be changed. For a given input
  tensor with vertex name `v``, its name in the output partition will be `(v, 1)`. Any
  delta tensor will have name `(v, 2)`, and any scalar tensor used to maintain the connectivity
  of the partition will have name `(v, 3)`.
Note: for a given binary tree with n indices, the output partition will contain 2n-1 vertices,
  with each leaf vertex corresponding to a sub tn adjacent to one output index. Keeping these
  leaf vertices in the partition makes later `approx_itensornetwork` algorithms more efficient.
Note: name of vertices in the output partition are the same as the name of vertices
  in `inds_btree`.
"""
function _partition(
  ::Algorithm"mincut_recursive_bisection", tn::ITensorNetwork, inds_btree::DataGraph
)
  @assert _is_rooted_directed_binary_tree(inds_btree)
  output_tns = Vector{ITensorNetwork}()
  output_deltas_vector = Vector{Vector{ITensor}}()
  scalars_vector = Vector{Vector{ITensor}}()
  # Mapping each vertex of the binary tree to a tn representing the partition
  # of the subtree containing this vertex and its descendant vertices.
  leaves = leaf_vertices(inds_btree)
  root = _root(inds_btree)
  v_to_subtree_tn = Dict{vertextype(inds_btree),ITensorNetwork}()
  v_to_subtree_tn[root] = disjoint_union(tn, ITensorNetwork())
  for v in pre_order_dfs_vertices(inds_btree, root)
    @assert haskey(v_to_subtree_tn, v)
    input_tn = v_to_subtree_tn[v]
    if !is_leaf(inds_btree, v)
      c1, c2 = child_vertices(inds_btree, v)
      descendant_c1 = pre_order_dfs_vertices(inds_btree, c1)
      indices = [inds_btree[l] for l in intersect(descendant_c1, leaves)]
      tn1, input_tn = _binary_partition(input_tn, indices)
      v_to_subtree_tn[c1] = tn1
      descendant_c2 = pre_order_dfs_vertices(inds_btree, c2)
      indices = [inds_btree[l] for l in intersect(descendant_c2, leaves)]
      tn1, input_tn = _binary_partition(input_tn, indices)
      v_to_subtree_tn[c2] = tn1
    end
    tn = rename_vertices(u -> u[1], subgraph(u -> u[2] == 1, input_tn))
    deltas = Vector{ITensor}(subgraph(u -> u[2] == 2, input_tn))
    scalars = Vector{ITensor}(subgraph(u -> u[2] == 3, input_tn))
    push!(output_tns, tn)
    push!(output_deltas_vector, deltas)
    push!(scalars_vector, scalars)
  end
  # In subgraph_vertices, each element is a vector of vertices to be
  # grouped in one partition.
  subgraph_vs = Vector{Vector{Tuple}}()
  delta_num = 0
  scalar_num = 0
  for (tn, deltas, scalars) in zip(output_tns, output_deltas_vector, scalars_vector)
    vs = Vector{Tuple}([(v, 1) for v in vertices(tn)])
    vs = vcat(vs, [(i + delta_num, 2) for i in 1:length(deltas)])
    vs = vcat(vs, [(i + scalar_num, 3) for i in 1:length(scalars)])
    push!(subgraph_vs, vs)
    delta_num += length(deltas)
    scalar_num += length(scalars)
  end
  out_tn = ITensorNetwork()
  for tn in output_tns
    for v in vertices(tn)
      add_vertex!(out_tn, v)
      out_tn[v] = tn[v]
    end
  end
  tn_deltas = ITensorNetwork(vcat(output_deltas_vector...))
  tn_scalars = ITensorNetwork(vcat(scalars_vector...))
  par = _partition(disjoint_union(out_tn, tn_deltas, tn_scalars), subgraph_vs)
  @assert is_tree(par)
  name_map = Dict()
  for (i, v) in enumerate(pre_order_dfs_vertices(inds_btree, root))
    name_map[i] = v
  end
  return rename_vertices(par, name_map)
end

function _partition(tn::ITensorNetwork, inds_btree::DataGraph; alg::String)
  return _partition(Algorithm(alg), tn, inds_btree)
end
