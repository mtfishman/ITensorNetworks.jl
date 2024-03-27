using DataGraphs: DataGraphs, AbstractDataGraph, edge_data, edge_data_type, vertex_data
using Graphs: Graphs
using ITensors: ITensors, unioninds, uniqueinds
using NamedGraphs: NamedGraphs, incident_edges, rename_vertices

abstract type AbstractIndsNetwork{V,I} <: AbstractDataGraph{V,Vector{I},Vector{I}} end

# Field access
data_graph(graph::AbstractIndsNetwork) = not_implemented()

# Overload if needed
Graphs.is_directed(::Type{<:AbstractIndsNetwork}) = false

# AbstractDataGraphs overloads
function DataGraphs.vertex_data(graph::AbstractIndsNetwork, args...)
  return vertex_data(data_graph(graph), args...)
end
function DataGraphs.edge_data(graph::AbstractIndsNetwork, args...)
  return edge_data(data_graph(graph), args...)
end

# TODO: Define a generic fallback for `AbstractDataGraph`?
edge_data_type(::Type{<:AbstractIndsNetwork{V,I}}) where {V,I} = Vector{I}

# 
# Index access
# 

function ITensors.uniqueinds(is::AbstractIndsNetwork, edge::AbstractEdge)
  inds = IndexSet(get(is, src(edge), Index[]))
  for ei in setdiff(incident_edges(is, src(edge)), [edge])
    inds = unioninds(inds, get(is, ei, Index[]))
  end
  return inds
end

function ITensors.uniqueinds(is::AbstractIndsNetwork, edge::Pair)
  return uniqueinds(is, edgetype(is)(edge))
end

function Base.union(tn1::AbstractIndsNetwork, tn2::AbstractIndsNetwork; kwargs...)
  return IndsNetwork(union(data_graph(tn1), data_graph(tn2); kwargs...))
end

function NamedGraphs.rename_vertices(f::Function, tn::AbstractIndsNetwork)
  return IndsNetwork(rename_vertices(f, data_graph(tn)))
end

# 
# Convenience functions
# 

function union_all_inds(is_in::AbstractIndsNetwork...)
  @assert all(map(ug -> ug == underlying_graph(is_in[1]), underlying_graph.(is_in)))
  is_out = IndsNetwork(underlying_graph(is_in[1]))
  for v in vertices(is_out)
    if any(isassigned(is, v) for is in is_in)
      is_out[v] = unioninds([get(is, v, Index[]) for is in is_in]...)
    end
  end
  for e in edges(is_out)
    if any(isassigned(is, e) for is in is_in)
      is_out[e] = unioninds([get(is, e, Index[]) for is in is_in]...)
    end
  end
  return is_out
end

function insert_missing_internal_inds(
  indsnetwork::AbstractIndsNetwork,
  edges=edges(indsnetwork);
  internal_inds_space=trivial_space(indsnetwork),
)
  indsnetwork = copy(indsnetwork)
  for e in edges
    if !isassigned(indsnetwork, e)
      iₑ = Index(internal_inds_space, edge_tag(e))
      indsnetwork[e] = [iₑ]
    end
  end
  return indsnetwork
end
