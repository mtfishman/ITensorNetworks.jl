using DataGraphs: DataGraph
using ITensors: ITensor
using NamedGraphs: NamedGraphs, NamedEdge, NamedGraph, vertextype

struct Private end

"""
    ITensorNetwork
"""
struct ITensorNetwork{V} <: AbstractITensorNetwork{V}
  data_graph::DataGraph{V,ITensor,ITensor,NamedGraph{V},NamedEdge{V}}
  function ITensorNetwork{V}(::Private, data_graph::DataGraph) where {V}
    return new{V}(data_graph)
  end
end

#
# Data access
#

data_graph(tn::ITensorNetwork) = getfield(tn, :data_graph)
data_graph_type(TN::Type{<:ITensorNetwork}) = fieldtype(TN, :data_graph)

function underlying_graph_type(TN::Type{<:ITensorNetwork})
  return fieldtype(data_graph_type(TN), :underlying_graph)
end

function ITensorNetwork{V}(data_graph::DataGraph{V}) where {V}
  return ITensorNetwork{V}(Private(), copy(data_graph))
end
function ITensorNetwork{V}(data_graph::DataGraph) where {V}
  return ITensorNetwork{V}(Private(), DataGraph{V}(data_graph))
end

ITensorNetwork(data_graph::DataGraph) = ITensorNetwork{vertextype(data_graph)}(data_graph)

function ITensorNetwork{V}() where {V}
  return ITensorNetwork{V}(data_graph_type(ITensorNetwork{V})())
end

ITensorNetwork() = ITensorNetwork{Any}()

# Conversion
ITensorNetwork(tn::ITensorNetwork) = copy(tn)
ITensorNetwork{V}(tn::ITensorNetwork{V}) where {V} = copy(tn)
function ITensorNetwork{V}(tn::AbstractITensorNetwork) where {V}
  return ITensorNetwork{V}(Private(), DataGraph{V}(data_graph(tn)))
end
ITensorNetwork(tn::AbstractITensorNetwork) = ITensorNetwork{vertextype(tn)}(tn)

NamedGraphs.convert_vertextype(::Type{V}, tn::ITensorNetwork{V}) where {V} = tn
NamedGraphs.convert_vertextype(V::Type, tn::ITensorNetwork) = ITensorNetwork{V}(tn)

Base.copy(tn::ITensorNetwork) = ITensorNetwork(copy(data_graph(tn)))

#
# Construction from collections of ITensors
#

ITensorNetwork(vs::Vector, ts::Vector{ITensor}) = ITensorNetwork(Dictionary(vs, ts))

ITensorNetwork(ts::Vector{<:Pair{<:Any,ITensor}}) = ITensorNetwork(dictionary(ts))

function ITensorNetwork(ts::ITensorCollection)
  return ITensorNetwork{keytype(ts)}(ts)
end

function ITensorNetwork{V}(ts::ITensorCollection) where {V}
  g = NamedGraph{V}(collect(eachindex(ts)))
  tn = ITensorNetwork(g)
  for v in vertices(g)
    tn[v] = ts[v]
  end
  return tn
end

function ITensorNetwork(t::ITensor)
  ts = ITensor[t]
  return ITensorNetwork{keytype(ts)}(ts)
end

#
# Construction from underyling named graph
#

function ITensorNetwork{V}(
  eltype::Type, undef::UndefInitializer, graph::AbstractNamedGraph; kwargs...
) where {V}
  return ITensorNetwork{V}(eltype, undef, IndsNetwork{V}(graph; kwargs...))
end

function ITensorNetwork{V}(eltype::Type, graph::AbstractNamedGraph; kwargs...) where {V}
  return ITensorNetwork{V}(eltype, IndsNetwork{V}(graph; kwargs...))
end

function ITensorNetwork{V}(
  undef::UndefInitializer, graph::AbstractNamedGraph; kwargs...
) where {V}
  return ITensorNetwork{V}(undef, IndsNetwork{V}(graph; kwargs...))
end

function ITensorNetwork{V}(graph::AbstractNamedGraph; kwargs...) where {V}
  return ITensorNetwork{V}(IndsNetwork{V}(graph; kwargs...))
end

function ITensorNetwork(
  eltype::Type, undef::UndefInitializer, graph::AbstractNamedGraph; kwargs...
)
  return ITensorNetwork{vertextype(graph)}(eltype, undef, graph; kwargs...)
end

function ITensorNetwork(eltype::Type, graph::AbstractNamedGraph; kwargs...)
  return ITensorNetwork{vertextype(graph)}(eltype, graph; kwargs...)
end

function ITensorNetwork(undef::UndefInitializer, graph::AbstractNamedGraph; kwargs...)
  return ITensorNetwork{vertextype(graph)}(undef, graph; kwargs...)
end

function ITensorNetwork(graph::AbstractNamedGraph; kwargs...)
  return ITensorNetwork{vertextype(graph)}(graph; kwargs...)
end

function ITensorNetwork(
  itensor_constructor::Function, underlying_graph::AbstractNamedGraph; kwargs...
)
  return ITensorNetwork(itensor_constructor, IndsNetwork(underlying_graph; kwargs...))
end

#
# Construction from underyling simple graph
#

function ITensorNetwork(
  eltype::Type, undef::UndefInitializer, graph::AbstractSimpleGraph; kwargs...
)
  return ITensorNetwork(eltype, undef, NamedGraph(graph); kwargs...)
end

function ITensorNetwork(eltype::Type, graph::AbstractSimpleGraph; kwargs...)
  return ITensorNetwork(eltype, NamedGraph(graph); kwargs...)
end

function ITensorNetwork(undef::UndefInitializer, graph::AbstractSimpleGraph; kwargs...)
  return ITensorNetwork(undef, NamedGraph(graph); kwargs...)
end

function ITensorNetwork(graph::AbstractSimpleGraph; kwargs...)
  return ITensorNetwork(NamedGraph(graph); kwargs...)
end

function ITensorNetwork(
  itensor_constructor::Function, underlying_graph::AbstractSimpleGraph; kwargs...
)
  return ITensorNetwork(itensor_constructor, NamedGraph(underlying_graph); kwargs...)
end

#
# Construction from IndsNetwork
#

function ITensorNetwork{V}(
  eltype::Type, undef::UndefInitializer, inds_network::IndsNetwork; kwargs...
) where {V}
  return ITensorNetwork{V}(inds_network; kwargs...) do v, inds...
    return ITensor(eltype, undef, inds...)
  end
end

function ITensorNetwork{V}(eltype::Type, inds_network::IndsNetwork; kwargs...) where {V}
  return ITensorNetwork{V}(inds_network; kwargs...) do v, inds...
    return ITensor(eltype, inds...)
  end
end

function ITensorNetwork{V}(
  undef::UndefInitializer, inds_network::IndsNetwork; kwargs...
) where {V}
  return ITensorNetwork{V}(inds_network; kwargs...) do v, inds...
    return ITensor(undef, inds...)
  end
end

function ITensorNetwork{V}(inds_network::IndsNetwork; kwargs...) where {V}
  return ITensorNetwork{V}(inds_network; kwargs...) do v, inds...
    return ITensor(inds...)
  end
end

function ITensorNetwork{V}(
  itensor_constructor::Function, inds_network::IndsNetwork; link_space=1, kwargs...
) where {V}
  # Graphs.jl uses `zero` to create a graph of the same type
  # without any vertices or edges.
  inds_network_merge = typeof(inds_network)(
    underlying_graph(inds_network); link_space, kwargs...
  )
  inds_network = union(inds_network_merge, inds_network)
  tn = ITensorNetwork{V}()
  for v in vertices(inds_network)
    add_vertex!(tn, v)
  end
  for e in edges(inds_network)
    add_edge!(tn, e)
  end
  for v in vertices(tn)
    siteinds = get(inds_network, v, indtype(inds_network)[])
    linkinds = [
      get(inds_network, edgetype(inds_network)(v, nv), indtype(inds_network)[]) for
      nv in neighbors(inds_network, v)
    ]
    setindex_preserve_graph!(tn, itensor_constructor(v, siteinds, linkinds...), v)
  end
  return tn
end

function ITensorNetwork(inds_network::IndsNetwork; kwargs...)
  return ITensorNetwork{vertextype(inds_network)}(inds_network; kwargs...)
end

function ITensorNetwork(
  eltype::Type, undef::UndefInitializer, inds_network::IndsNetwork; kwargs...
)
  return ITensorNetwork{vertextype(inds_network)}(eltype, undef, inds_network; kwargs...)
end

function ITensorNetwork(eltype::Type, inds_network::IndsNetwork; kwargs...)
  return ITensorNetwork{vertextype(inds_network)}(eltype, inds_network; kwargs...)
end

function ITensorNetwork(undef::UndefInitializer, inds_network::IndsNetwork; kwargs...)
  return ITensorNetwork{vertextype(inds_network)}(undef, inds_network; kwargs...)
end

function ITensorNetwork(itensor_constructor::Function, inds_network::IndsNetwork; kwargs...)
  return ITensorNetwork{vertextype(inds_network)}(
    itensor_constructor, inds_network; kwargs...
  )
end

# TODO: Deprecate in favor of version above? Or use keyword argument?
# This can be handled by `ITensorNetwork((v, inds...) -> state(inds...), inds_network)`
function ITensorNetwork(eltype::Type, is::IndsNetwork, initstate::Function)
  ψ = ITensorNetwork(eltype, is)
  for v in vertices(ψ)
    ψ[v] = convert_eltype(eltype, state(initstate(v), only(is[v])))
  end
  ψ = insert_links(ψ, edges(is))
  return ψ
end

function ITensorNetwork(eltype::Type, is::IndsNetwork, initstate::Union{String,Integer})
  return ITensorNetwork(eltype, is, v -> initstate)
end

function ITensorNetwork(is::IndsNetwork, initstate::Union{String,Integer,Function})
  return ITensorNetwork(Number, is, initstate)
end

function insert_links(ψ::ITensorNetwork, edges::Vector=edges(ψ); cutoff=1e-15)
  for e in edges
    # Define this to work?
    # ψ = factorize(ψ, e; cutoff)
    ψᵥ₁, ψᵥ₂ = factorize(ψ[src(e)] * ψ[dst(e)], inds(ψ[src(e)]); cutoff, tags=edge_tag(e))
    ψ[src(e)] = ψᵥ₁
    ψ[dst(e)] = ψᵥ₂
  end
  return ψ
end

ITensorNetwork(itns::Vector{ITensorNetwork}) = reduce(⊗, itns)

function Base.Vector{ITensor}(ψ::ITensorNetwork)
  return ITensor[ψ[v] for v in vertices(ψ)]
end
