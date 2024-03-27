"""
    TreeTensorNetwork{V} <: AbstractTreeTensorNetwork{V}

# Fields

- itensor_network::ITensorNetwork{V}
- ortho_lims::Vector{V}: A vector of vertices defining the orthogonality limits.

"""
struct TreeTensorNetwork{V} <: AbstractTreeTensorNetwork{V}
  itensor_network::ITensorNetwork{V}
  ortho_center::Vector{V}
  function TreeTensorNetwork{V}(
    itensor_network::ITensorNetwork, ortho_center::Vector=vertices(itensor_network)
  ) where {V}
    @assert is_tree(itensor_network)
    return new{V}(itensor_network, ortho_center)
  end
end

const TTN = TreeTensorNetwork

function data_graph_type(G::Type{<:TTN})
  return data_graph_type(fieldtype(G, :itensor_network))
end

function Base.copy(ψ::TTN)
  return TTN(copy(ψ.itensor_network), copy(ψ.ortho_center))
end

# Field access
itensor_network(ψ::TTN) = getfield(ψ, :itensor_network)

# Required for `AbstractITensorNetwork` interface
data_graph(ψ::TTN) = data_graph(itensor_network(ψ))

# 
# Constructor
# 

TTN(tn::ITensorNetwork, args...) = TTN{vertextype(tn)}(tn, args...)

# catch-all for default ElType
function TTN(g::AbstractGraph, args...; kwargs...)
  return TTN(Float64, g, args...; kwargs...)
end

function TTN(eltype::Type{<:Number}, graph::AbstractGraph, args...; kwargs...)
  itensor_network = ITensorNetwork(eltype, graph; kwargs...)
  return TTN(itensor_network, args...)
end

# construct from given state (map)
function TTN(::Type{ElT}, is::AbstractIndsNetwork, initstate, args...) where {ElT<:Number}
  itensor_network = ITensorNetwork(ElT, is, initstate)
  return TTN(itensor_network, args...)
end

# Constructor from a collection of ITensors.
# TODO: Support other collections like `Dictionary`,
# interface for custom vertex names.
function TTN(ts::ITensorCollection)
  return TTN(ITensorNetwork(ts))
end

# TODO: Implement `random_circuit_ttn` for non-trivial
# bond dimensions and correlations.
# TODO: Implement random_ttn for QN-Index
function random_ttn(args...; kwargs...)
  T = TTN(args...; kwargs...)
  randn!.(vertex_data(T))
  normalize!.(vertex_data(T))
  return T
end

function random_mps(
  external_inds::Vector{<:Index};
  states=nothing,
  internal_inds_space=trivial_space(external_inds),
)
  # TODO: Implement in terms of general
  # `random_ttn` constructor
  tn_mps = if isnothing(states)
    randomMPS(external_inds; linkdims=internal_inds_space)
  else
    randomMPS(external_inds, states; linkdims=internal_inds_space)
  end
  return TTN([tn_mps[v] for v in eachindex(tn_mps)])
end

#
# Construction from operator (map)
#

function TTN(
  ::Type{ElT},
  sites_map::Pair{<:AbstractIndsNetwork,<:AbstractIndsNetwork},
  ops::Dictionary;
  kwargs...,
) where {ElT<:Number}
  s = first(sites_map) # TODO: Use the sites_map
  N = nv(sites)
  os = Prod{Op}()
  for v in vertices(sites)
    os *= Op(ops[v], v)
  end
  T = TTN(ElT, os, sites; kwargs...)
  # see https://github.com/ITensor/ITensors.jl/issues/526
  lognormT = lognorm(T)
  T /= exp(lognormT / N) # TODO: fix broadcasting for in-place assignment
  truncate!(T; cutoff=1e-15)
  T *= exp(lognormT / N)
  return T
end

function TTN(
  ::Type{ElT},
  sites_map::Pair{<:AbstractIndsNetwork,<:AbstractIndsNetwork},
  fops::Function;
  kwargs...,
) where {ElT<:Number}
  sites = first(sites_map) # TODO: Use the sites_map
  ops = Dictionary(vertices(sites), map(v -> fops(v), vertices(sites)))
  return TTN(ElT, sites, ops; kwargs...)
end

function TTN(
  ::Type{ElT},
  sites_map::Pair{<:AbstractIndsNetwork,<:AbstractIndsNetwork},
  op::String;
  kwargs...,
) where {ElT<:Number}
  sites = first(sites_map) # TODO: Use the sites_map
  ops = Dictionary(vertices(sites), fill(op, nv(sites)))
  return TTN(ElT, sites, ops; kwargs...)
end

# Special constructors

function mps(external_inds::Vector{<:Index}; states)
  return mps(map(i -> [i], external_inds); states)
end

function mps(external_inds::Vector{<:Vector{<:Index}}; states)
  g = named_path_graph(length(external_inds))
  tn = ITensorNetwork(g)
  for v in vertices(tn)
    tn[v] = state(only(external_inds[v]), states(v))
  end
  tn = insert_missing_internal_inds(
    tn, edges(g); internal_inds_space=trivial_space(indtype(external_inds))
  )
  return TTN(tn)
end

## function mps(external_inds::Vector{<:Index}; states)
##   is = path_indsnetwork(external_inds)
##   tn = TTN(underlying_graph(is))
##   tn = insert_missing_internal_inds(tn, trivial_space(indtype(is)))
##   for v in vertices(tn)
##     @show v
##     @show tn[v]
##     tn[v] *= state(only(is[v]), states(v))
##     @show tn[v]
##   end
##   return tn
## end

## function productTTN(args...; kwargs...)
##   return TTN(args...; link_space=1, kwargs...)
## end

# 
# Utility
# 

function replacebond!(T::TTN, edge::AbstractEdge, phi::ITensor; kwargs...)
  ortho::String = get(kwargs, :ortho, "left")
  swapsites::Bool = get(kwargs, :swapsites, false)
  which_decomp::Union{String,Nothing} = get(kwargs, :which_decomp, nothing)
  normalize::Bool = get(kwargs, :normalize, false)

  indsTe = inds(T[src(edge)])
  if swapsites
    sb = siteinds(M, src(edge))
    sbp1 = siteinds(M, dst(edge))
    indsTe = replaceinds(indsTe, sb, sbp1)
  end

  L, R, spec = factorize(
    phi, indsTe; which_decomp=which_decomp, tags=tags(T, edge), kwargs...
  )

  T[src(edge)] = L
  T[dst(edge)] = R
  if ortho == "left"
    normalize && (T[dst(edge)] ./= norm(T[dst(edge)]))
    isortho(T) && (T = set_ortho_center(T, [dst(edge)]))
  elseif ortho == "right"
    normalize && (T[src(edge)] ./= norm(T[src(edge)]))
    isortho(T) && (T = set_ortho_center(T, [src(edge)]))
  end
  return spec
end

function replacebond!(T::TTN, edge::Pair, phi::ITensor; kwargs...)
  return replacebond!(T, edgetype(T)(edge), phi; kwargs...)
end

function replacebond(T0::TTN, args...; kwargs...)
  return replacebond!(copy(T0), args...; kwargs...)
end
