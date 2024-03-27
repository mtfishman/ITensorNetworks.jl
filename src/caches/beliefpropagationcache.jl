using NamedGraphs: boundary_partitionedges

default_message(inds_e) = ITensor[denseblocks(delta(inds_e))]
default_messages(ptn::PartitionedGraph) = Dictionary()
function default_message_update(contract_list::Vector{ITensor}; kwargs...)
  sequence = optimal_contraction_sequence(contract_list)
  updated_messages = contract(contract_list; sequence, kwargs...)
  updated_messages /= norm(updated_messages)
  return ITensor[updated_messages]
end
@traitfn default_bp_maxiter(g::::(!IsDirected)) = is_tree(g) ? 1 : nothing
@traitfn function default_bp_maxiter(g::::IsDirected)
  return default_bp_maxiter(undirected_graph(underlying_graph(g)))
end
default_partitioned_vertices(ψ::AbstractITensorNetwork) = group(v -> v, vertices(ψ))
default_cache_update_kwargs(cache) = (; maxiter=20, tol=1e-5)

function message_diff(message_a::Vector{ITensor}, message_b::Vector{ITensor})
  lhs, rhs = contract(message_a), contract(message_b)
  return 0.5 *
         norm((denseblocks(lhs) / sum(diag(lhs))) - (denseblocks(rhs) / sum(diag(rhs))))
end

struct BeliefPropagationCache{PTN,MTS,DM}
  partitioned_itensornetwork::PTN
  messages::MTS
  default_message::DM
end

#Constructors...
function BeliefPropagationCache(
  ptn::PartitionedGraph; messages=default_messages(ptn), default_message=default_message
)
  return BeliefPropagationCache(ptn, messages, default_message)
end

function BeliefPropagationCache(tn, partitioned_vertices; kwargs...)
  ptn = PartitionedGraph(tn, partitioned_vertices)
  return BeliefPropagationCache(ptn; kwargs...)
end

function BeliefPropagationCache(tn; kwargs...)
  return BeliefPropagationCache(tn, default_partitioning(tn); kwargs...)
end

function partitioned_itensornetwork(bp_cache::BeliefPropagationCache)
  return bp_cache.partitioned_itensornetwork
end
messages(bp_cache::BeliefPropagationCache) = bp_cache.messages
default_message(bp_cache::BeliefPropagationCache) = bp_cache.default_message
function tensornetwork(bp_cache::BeliefPropagationCache)
  return unpartitioned_graph(partitioned_itensornetwork(bp_cache))
end

#Forward from partitioned graph
for f in [
  :(NamedGraphs.partitioned_graph),
  :(NamedGraphs.partitionedge),
  :(NamedGraphs.partitionvertices),
  :(NamedGraphs.vertices),
  :(NamedGraphs.boundary_partitionedges),
  :linkinds,
]
  @eval begin
    function $f(bp_cache::BeliefPropagationCache, args...; kwargs...)
      return $f(partitioned_itensornetwork(bp_cache), args...; kwargs...)
    end
  end
end

function default_message(bp_cache::BeliefPropagationCache, edge::PartitionEdge)
  return default_message(bp_cache)(linkinds(bp_cache, edge))
end

function message(bp_cache::BeliefPropagationCache, edge::PartitionEdge)
  mts = messages(bp_cache)
  return get(mts, edge, default_message(bp_cache, edge))
end
function messages(
  bp_cache::BeliefPropagationCache, edges::Vector{<:PartitionEdge}; kwargs...
)
  return [message(bp_cache, edge; kwargs...) for edge in edges]
end

function Base.copy(bp_cache::BeliefPropagationCache)
  return BeliefPropagationCache(
    copy(partitioned_itensornetwork(bp_cache)),
    copy(messages(bp_cache)),
    default_message(bp_cache),
  )
end

function default_bp_maxiter(bp_cache::BeliefPropagationCache)
  return default_bp_maxiter(partitioned_graph(bp_cache))
end
function default_edge_sequence(bp_cache::BeliefPropagationCache)
  return default_edge_sequence(partitioned_itensornetwork(bp_cache))
end

function set_messages(cache::BeliefPropagationCache, messages)
  return BeliefPropagationCache(
    partitioned_itensornetwork(cache), messages, default_message(cache)
  )
end

function environment(
  bp_cache::BeliefPropagationCache,
  partition_vertices::Vector{<:PartitionVertex};
  ignore_edges=PartitionEdge[],
)
  bpes = boundary_partitionedges(bp_cache, partition_vertices; dir=:in)
  ms = messages(bp_cache, setdiff(bpes, ignore_edges))
  return reduce(vcat, ms; init=[])
end

function environment(
  bp_cache::BeliefPropagationCache, partition_vertex::PartitionVertex; kwargs...
)
  return environment(bp_cache, [partition_vertex]; kwargs...)
end

function environment(bp_cache::BeliefPropagationCache, verts::Vector)
  partition_verts = partitionvertices(bp_cache, verts)
  messages = environment(bp_cache, partition_verts)
  central_tensors = ITensor[
    tensornetwork(bp_cache)[v] for v in setdiff(vertices(bp_cache, partition_verts), verts)
  ]
  return vcat(messages, central_tensors)
end

function factor(bp_cache::BeliefPropagationCache, vertex::PartitionVertex)
  ptn = partitioned_itensornetwork(bp_cache)
  return Vector{ITensor}(subgraph(ptn, vertex))
end

"""
Compute message tensor as product of incoming mts and local state
"""
function update_message(
  bp_cache::BeliefPropagationCache,
  edge::PartitionEdge;
  message_update=default_message_update,
  message_update_kwargs=(;),
)
  vertex = src(edge)
  messages = environment(bp_cache, vertex; ignore_edges=PartitionEdge[reverse(edge)])
  state = factor(bp_cache, vertex)

  return message_update(ITensor[messages; state]; message_update_kwargs...)
end

"""
Do a sequential update of the message tensors on `edges`
"""
function update(
  bp_cache::BeliefPropagationCache,
  edges::Vector{<:PartitionEdge};
  (update_diff!)=nothing,
  kwargs...,
)
  bp_cache_updated = copy(bp_cache)
  mts = messages(bp_cache_updated)
  for e in edges
    set!(mts, e, update_message(bp_cache_updated, e; kwargs...))
    if !isnothing(update_diff!)
      update_diff![] += message_diff(message(bp_cache, e), mts[e])
    end
  end
  return bp_cache_updated
end

"""
Update the message tensor on a single edge
"""
function update(bp_cache::BeliefPropagationCache, edge::PartitionEdge; kwargs...)
  return update(bp_cache, [edge]; kwargs...)
end

"""
Do parallel updates between groups of edges of all message tensors
Currently we send the full message tensor data struct to update for each edge_group. But really we only need the
mts relevant to that group.
"""
function update(
  bp_cache::BeliefPropagationCache,
  edge_groups::Vector{<:Vector{<:PartitionEdge}};
  kwargs...,
)
  new_mts = copy(messages(bp_cache))
  for edges in edge_groups
    bp_cache_t = update(bp_cache, edges; kwargs...)
    for e in edges
      new_mts[e] = message(bp_cache_t, e)
    end
  end
  return set_messages(bp_cache, new_mts)
end

"""
More generic interface for update, with default params
"""
function update(
  bp_cache::BeliefPropagationCache;
  edges=default_edge_sequence(bp_cache),
  maxiter=default_bp_maxiter(bp_cache),
  tol=nothing,
  verbose=false,
  kwargs...,
)
  compute_error = !isnothing(tol)
  diff = compute_error ? Ref(0.0) : nothing
  if isnothing(maxiter)
    error("You need to specify a number of iterations for BP!")
  end
  for i in 1:maxiter
    bp_cache = update(bp_cache, edges; (update_diff!)=diff, kwargs...)
    if compute_error && (diff.x / length(edges)) <= tol
      if verbose
        println("BP converged to desired precision after $i iterations.")
      end
      break
    end
  end
  return bp_cache
end

"""
Update the tensornetwork inside the cache
"""
function update_factors(
  bp_cache::BeliefPropagationCache, vertices::Vector, factors::Vector{ITensor}
)
  bp_cache = copy(bp_cache)
  tn = tensornetwork(bp_cache)

  for (vertex, factor) in zip(vertices, factors)
    # TODO: Add a check that this preserves the graph structure.
    setindex_preserve_graph!(tn, factor, vertex)
  end
  return bp_cache
end

function update_factor(bp_cache, vertex, factor)
  return update_factors(bp_cache, [vertex], ITensor[factor])
end
