using DataGraphs: DataGraphs, vertex_data
using Graphs: Graphs
using ITensors: Index, dag
using ITensors.ITensorVisualizationCore: ITensorVisualizationCore, visualize
using NamedGraphs: NamedGraphs, NamedEdge, NamedGraph, vertextype

struct IndsNetwork{V,I} <: AbstractIndsNetwork{V,I}
  data_graph::DataGraph{V,Vector{I},Vector{I},NamedGraph{V},NamedEdge{V}}
  global function _IndsNetwork(V::Type, I::Type, g::DataGraph)
    return new{V,I}(g)
  end
end
indtype(inds_network::IndsNetwork) = indtype(typeof(inds_network))
indtype(::Type{<:IndsNetwork{V,I}}) where {V,I} = I
data_graph(is::IndsNetwork) = is.data_graph
DataGraphs.underlying_graph(is::IndsNetwork) = underlying_graph(data_graph(is))
NamedGraphs.vertextype(::Type{<:IndsNetwork{V}}) where {V} = V
DataGraphs.underlying_graph_type(G::Type{<:IndsNetwork}) = NamedGraph{vertextype(G)}
Graphs.is_directed(::Type{<:IndsNetwork}) = false

#
# Constructor
#

# When setting an edge with collections of `Index`, set the reverse direction
# edge with the `dag`.
function reverse_data_direction(
  inds_network::IndsNetwork, is::Union{Index,Tuple{Vararg{Index}},Vector{<:Index}}
)
  return dag(is)
end

function IndsNetwork{V}(data_graph::DataGraph) where {V}
  I = eltype(eltype(vertex_data(data_graph)))
  return IndsNetwork{V,I}(data_graph)
end

function IndsNetwork(data_graph::DataGraph)
  return IndsNetwork{vertextype(data_graph)}(data_graph)
end

@traitfn function IndsNetwork{V,I}(g::G) where {G<:DataGraph,V,I;!IsUnderlyingGraph{G}}
  return _IndsNetwork(V, I, g)
end

function IndsNetwork{V,I}(g::AbstractNamedGraph, link_space, site_space) where {V,I}
  link_space_dictionary = link_space_map(V, I, g, link_space)
  site_space_dictionary = site_space_map(V, I, g, site_space)
  return IndsNetwork{V,I}(g, link_space_dictionary, site_space_dictionary)
end

function IndsNetwork{V,I}(g::AbstractSimpleGraph, link_space, site_space) where {V,I}
  return IndsNetwork{V,I}(NamedGraph(g), link_space, site_space)
end

@traitfn function IndsNetwork{V}(
  g::G, link_space, site_space
) where {G,V;IsUnderlyingGraph{G}}
  I = indtype(link_space, site_space)
  return IndsNetwork{V,I}(g, link_space, site_space)
end

@traitfn function IndsNetwork(g::G, link_space, site_space) where {G; IsUnderlyingGraph{G}}
  V = vertextype(g)
  return IndsNetwork{V}(g, link_space, site_space)
end

# Core constructor, others should convert
# their inputs to these types.
# TODO: Make this an inner constructor `_IndsNetwork`?
function IndsNetwork{V,I}(
  g::AbstractNamedGraph,
  link_space::Dictionary{<:Any,<:Vector{<:Index}},
  site_space::Dictionary{<:Any,<:Vector{<:Index}},
) where {V,I}
  dg = DataGraph{V,Vector{I},Vector{I}}(g)
  for e in keys(link_space)
    dg[e] = link_space[e]
  end
  for v in keys(site_space)
    dg[v] = site_space[v]
  end
  return IndsNetwork{V}(dg)
end

function IndsNetwork{V,I}(
  g::AbstractSimpleGraph,
  link_space::Dictionary{<:Any,<:Vector{<:Index}},
  site_space::Dictionary{<:Any,<:Vector{<:Index}},
) where {V,I}
  return IndsNetwork{V,I}(NamedGraph(g), link_space, site_space)
end

# Construct from collections of indices

function path_indsnetwork(external_inds::Vector{<:Vector{<:Index}})
  g = named_path_graph(length(external_inds))
  is = IndsNetwork(g)
  for v in eachindex(external_inds)
    is[v] = external_inds[v]
  end
  return is
end

function path_indsnetwork(external_inds::Vector{<:Index})
  return path_indsnetwork(map(i -> [i], external_inds))
end

# TODO: Replace with a trait of the same name.
const IsIndexSpace = Union{<:Integer,Vector{<:Pair{QN,<:Integer}}}

# Infer the `Index` type of an `IndsNetwork` from the
# spaces that get input.
indtype(link_space::Nothing, site_space::Nothing) = Index
indtype(link_space::Nothing, site_space) = indtype(site_space)
indtype(link_space, site_space::Nothing) = indtype(link_space)
indtype(link_space, site_space) = promote_type(indtype(link_space), indtype(site_space))

# Default to type space
indtype(space) = _indtype(typeof(space))

# Base case
# Use `_indtype` to avoid recursion overflow
_indtype(T::Type{<:Index}) = T
_indtype(T::Type{<:IsIndexSpace}) = Index{T}
_indtype(::Type{Nothing}) = Index

# Containers
_indtype(T::Type{<:AbstractDictionary}) = _indtype(eltype(T))
_indtype(T::Type{<:AbstractVector}) = _indtype(eltype(T))

@traitfn function default_link_space(V::Type, g::::IsUnderlyingGraph)
  # TODO: Convert `g` to vertex type `V`
  E = edgetype(g)
  return Dictionary{E,Vector{Index}}()
end

@traitfn function default_site_space(V::Type, g::::IsUnderlyingGraph)
  return Dictionary{V,Vector{Index}}()
end

@traitfn function IndsNetwork{V,I}(
  g::G; link_space=nothing, site_space=nothing
) where {G,V,I;IsUnderlyingGraph{G}}
  return IndsNetwork{V,I}(g, link_space, site_space)
end

@traitfn function IndsNetwork{V}(
  g::G; link_space=nothing, site_space=nothing
) where {G,V;IsUnderlyingGraph{G}}
  return IndsNetwork{V}(g, link_space, site_space)
end

@traitfn function IndsNetwork(g::G; kwargs...) where {G; IsUnderlyingGraph{G}}
  return IndsNetwork{vertextype(g)}(g; kwargs...)
end

@traitfn function link_space_map(
  V::Type,
  I::Type{<:Index},
  g::::IsUnderlyingGraph,
  link_spaces::Dictionary{<:Any,Vector{Int}},
)
  # TODO: Convert `g` to vertex type `V`
  # @assert vertextype(g) == V
  E = edgetype(g)
  linkinds_dictionary = Dictionary{E,Vector{I}}()
  for e in keys(link_spaces)
    l = [edge_index(e, link_space) for link_space in link_spaces[e]]
    set!(linkinds_dictionary, e, l)
    # set!(linkinds_dictionary, reverse(e), dag(l))
  end
  return linkinds_dictionary
end

@traitfn function link_space_map(
  V::Type,
  I::Type{<:Index},
  g::::IsUnderlyingGraph,
  linkinds::AbstractDictionary{<:Any,Vector{<:Index}},
)
  E = edgetype(g)
  return convert(Dictionary{E,Vector{I}}, linkinds)
end

@traitfn function link_space_map(
  V::Type,
  I::Type{<:Index},
  g::::IsUnderlyingGraph,
  linkinds::AbstractDictionary{<:Any,<:Index},
)
  return link_space_map(V, I, g, map(l -> [l], linkinds))
end

@traitfn function link_space_map(
  V::Type, I::Type{<:Index}, g::::IsUnderlyingGraph, link_spaces::Dictionary{<:Any,Int}
)
  return link_space_map(V, I, g, map(link_space -> [link_space], link_spaces))
end

@traitfn function link_space_map(
  V::Type, I::Type{<:Index}, g::::IsUnderlyingGraph, link_spaces::Vector{Int}
)
  return link_space_map(V, I, g, map(Returns(link_spaces), Indices(edges(g))))
end

# TODO: Generalize using `IsIndexSpace` trait that is true for
# `Integer` and `Vector{<:Pair{QN,<:Integer}}`.
@traitfn function link_space_map(
  V::Type, I::Type{<:Index}, g::::IsUnderlyingGraph, link_space::Int
)
  return link_space_map(V, I, g, [link_space])
end

@traitfn function link_space_map(
  V::Type, I::Type{<:Index}, g::::IsUnderlyingGraph, link_space::Nothing
)
  # TODO: Make sure `edgetype(g)` is consistent with vertex type `V`
  return Dictionary{edgetype(g),Vector{I}}()
end

# TODO: Convert the dictionary according to `V` and `I`
@traitfn function link_space_map(
  V::Type,
  I::Type{<:Index},
  g::::IsUnderlyingGraph,
  link_space::AbstractDictionary{<:Any,<:Vector{<:Index}},
)
  return link_space
end

@traitfn function site_space_map(
  V::Type,
  I::Type{<:Index},
  g::::IsUnderlyingGraph,
  siteinds::AbstractDictionary{<:Any,Vector{<:Index}},
)
  return convert(Dictionary{V,Vector{I}}, siteinds)
end

@traitfn function site_space_map(
  V::Type,
  I::Type{<:Index},
  g::::IsUnderlyingGraph,
  siteinds::AbstractDictionary{<:Any,<:Index},
)
  return site_space_map(V, I, g, map(s -> [s], siteinds))
end

@traitfn function site_space_map(
  V::Type,
  I::Type{<:Index},
  g::::IsUnderlyingGraph,
  site_spaces::AbstractDictionary{<:Any,Vector{Int}},
)
  siteinds_dictionary = Dictionary{V,Vector{I}}()
  for v in keys(site_spaces)
    s = [vertex_index(v, site_space) for site_space in site_spaces[v]]
    set!(siteinds_dictionary, v, s)
  end
  return siteinds_dictionary
end

# TODO: Generalize using `IsIndexSpace` trait that is true for
# `Integer` and `Vector{<:Pair{QN,<:Integer}}`.
@traitfn function site_space_map(
  V::Type, I::Type{<:Index}, g::::IsUnderlyingGraph, site_space::Int
)
  return site_space_map(V, I, g, [site_space])
end

# Multiple site indices per vertex
# TODO: How to distinguish from block indices?
@traitfn function site_space_map(
  V::Type, I::Type{<:Index}, g::::IsUnderlyingGraph, site_spaces::Vector{Int}
)
  return site_space_map(V, I, g, map(Returns(site_spaces), Indices(vertices(g))))
end

@traitfn function site_space_map(
  V::Type,
  I::Type{<:Index},
  g::::IsUnderlyingGraph,
  site_spaces::AbstractDictionary{<:Any,Int},
)
  return site_space_map(V, I, g, map(site_space -> [site_space], site_spaces))
end

# TODO: Convert the dictionary according to `V` and `I`
@traitfn function site_space_map(
  V::Type,
  I::Type{<:Index},
  g::::IsUnderlyingGraph,
  site_space::AbstractDictionary{<:Any,<:Vector{<:Index}},
)
  return site_space
end

@traitfn function site_space_map(
  V::Type, I::Type{<:Index}, g::::IsUnderlyingGraph, site_space::Nothing
)
  return Dictionary{V,Vector{I}}()
end

#
# Utility
#

Base.copy(is::IndsNetwork) = IndsNetwork(copy(data_graph(is)))

function map_inds(f, is::IndsNetwork, args...; sites=nothing, links=nothing, kwargs...)
  return map_data(i -> f(i, args...; kwargs...), is; vertices=sites, edges=links)
end

#
# Visualization
#

function ITensorVisualizationCore.visualize(is::IndsNetwork, args...; kwargs...)
  return visualize(ITensorNetwork(is), args...; kwargs...)
end
