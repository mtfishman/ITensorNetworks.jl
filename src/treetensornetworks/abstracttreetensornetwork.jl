using TupleTools: TupleTools

abstract type AbstractTreeTensorNetwork{V} <: AbstractITensorNetwork{V} end

const AbstractTTN = AbstractTreeTensorNetwork

function underlying_graph_type(G::Type{<:AbstractTTN})
  return underlying_graph_type(data_graph_type(G))
end

# 
# Field access
# 

ITensorNetwork(ψ::AbstractTTN) = ψ.itensor_network
ortho_center(ψ::AbstractTTN) = ψ.ortho_center

function default_root_vertex(gs::AbstractGraph...)
  # @assert all(is_tree.(gs))
  return first(leaf_vertices(gs[end]))
end

# 
# Orthogonality center
# 

isortho(ψ::AbstractTTN) = isone(length(ortho_center(ψ)))

function set_ortho_center(ψ::AbstractTTN{V}, new_center::Vector{<:V}) where {V}
  return typeof(ψ)(itensor_network(ψ), new_center)
end

reset_ortho_center(ψ::AbstractTTN) = set_ortho_center(ψ, vertices(ψ))

# 
# Dense constructors
# 

# construct from dense ITensor, using IndsNetwork of site indices
function (::Type{TTNT})(
  A::ITensor, is::IndsNetwork; ortho_center=default_root_vertex(is), kwargs...
) where {TTNT<:AbstractTTN}
  for v in vertices(is)
    @assert hasinds(A, is[v])
  end
  @assert ortho_center ∈ vertices(is)
  ψ = ITensorNetwork(is)
  Ã = A
  for e in post_order_dfs_edges(ψ, ortho_center)
    left_inds = uniqueinds(is, e)
    L, R = factorize(Ã, left_inds; tags=edge_tag(e), ortho="left", kwargs...)
    l = commonind(L, R)
    ψ[src(e)] = L
    is[e] = [l]
    Ã = R
  end
  ψ[ortho_center] = Ã
  T = TTNT(ψ)
  T = orthogonalize(T, ortho_center)
  return T
end

# construct from dense ITensor, using AbstractNamedGraph and vector of site indices
# TODO: remove if it doesn't turn out to be useful
function (::Type{TTNT})(
  A::ITensor, sites::Vector, g::AbstractNamedGraph; vertex_order=vertices(g), kwargs...
) where {TTNT<:AbstractTTN}
  is = IndsNetwork(g; site_space=Dictionary(vertex_order, sites))
  return TTNT(A, is; kwargs...)
end

# construct from dense array, using IndsNetwork
# TODO: probably remove this one, doesn't seem very useful
function (::Type{TTNT})(
  A::AbstractArray{<:Number}, is::IndsNetwork; vertex_order=vertices(is), kwargs...
) where {TTNT<:AbstractTTN}
  sites = [is[v] for v in vertex_order]
  return TTNT(itensor(A, sites...), is; kwargs...)
end

# construct from dense array, using NamedDimGraph and vector of site indices
function (::Type{TTNT})(
  A::AbstractArray{<:Number}, sites::Vector, args...; kwargs...
) where {TTNT<:AbstractTTN}
  return TTNT(itensor(A, sites...), sites, args...; kwargs...)
end

# 
# Orthogonalization
# 

function orthogonalize(ψ::AbstractTTN{V}, root_vertex::V; kwargs...) where {V}
  (isortho(ψ) && only(ortho_center(ψ)) == root_vertex) && return ψ
  if isortho(ψ)
    edge_list = edge_path(ψ, only(ortho_center(ψ)), root_vertex)
  else
    edge_list = post_order_dfs_edges(ψ, root_vertex)
  end
  for e in edge_list
    ψ = orthogonalize(ψ, e)
  end
  return set_ortho_center(ψ, [root_vertex])
end

# For ambiguity error

function orthogonalize(tn::AbstractTTN, edge::AbstractEdge; kwargs...)
  return typeof(tn)(orthogonalize(ITensorNetwork(tn), edge; kwargs...))
end

# 
# Truncation
# 

function truncate(ψ::AbstractTTN; root_vertex=default_root_vertex(ψ), kwargs...)
  for e in post_order_dfs_edges(ψ, root_vertex)
    # always orthogonalize towards source first to make truncations controlled
    ψ = orthogonalize(ψ, src(e))
    ψ = truncate(ψ, e; kwargs...)
    ψ = set_ortho_center(ψ, [dst(e)])
  end
  return ψ
end

# For ambiguity error
function truncate(tn::AbstractTTN, edge::AbstractEdge; kwargs...)
  return typeof(tn)(truncate(ITensorNetwork(tn), edge; kwargs...))
end

#
# Contraction
#

# TODO: decide on contraction order: reverse dfs vertices or forward dfs edges?
function contract(
  ψ::AbstractTTN{V}, root_vertex::V=default_root_vertex(ψ); kwargs...
) where {V}
  ψ = copy(ψ)
  # reverse post order vertices
  traversal_order = reverse(post_order_dfs_vertices(ψ, root_vertex))
  return contract(ITensorNetwork(ψ); sequence=traversal_order, kwargs...)
  # # forward post order edges
  # ψ = copy(ψ)
  # for e in post_order_dfs_edges(ψ, root_vertex)
  #   ψ = contract(ψ, e)
  # end
  # return ψ[root_vertex]
end

function inner(ϕ::AbstractTTN, ψ::AbstractTTN; root_vertex=default_root_vertex(ϕ, ψ))
  ϕᴴ = sim(dag(ϕ); sites=[])
  ψ = sim(ψ; sites=[])
  ϕψ = ϕᴴ ⊗ ψ
  # TODO: find the largest tensor and use it as
  # the `root_vertex`.
  for e in post_order_dfs_edges(ψ, root_vertex)
    if has_vertex(ϕψ, (src(e), 2))
      ϕψ = contract(ϕψ, (src(e), 2) => (src(e), 1))
    end
    ϕψ = contract(ϕψ, (src(e), 1) => (dst(e), 1))
    if has_vertex(ϕψ, (dst(e), 2))
      ϕψ = contract(ϕψ, (dst(e), 2) => (dst(e), 1))
    end
  end
  return ϕψ[root_vertex, 1][]
end

function norm(ψ::AbstractTTN)
  if isortho(ψ)
    return norm(ψ[only(ortho_center(ψ))])
  end
  return √(abs(real(inner(ψ, ψ))))
end

# 
# Utility
# 

function normalize!(ψ::AbstractTTN)
  c = ortho_center(ψ)
  lognorm_ψ = lognorm(ψ)
  if lognorm_ψ == -Inf
    return ψ
  end
  z = exp(lognorm_ψ / length(c))
  for v in c
    ψ[v] ./= z
  end
  return ψ
end

function normalize(ψ::AbstractTTN)
  return normalize!(copy(ψ))
end

function _apply_to_orthocenter!(f, ψ::AbstractTTN, x)
  v = first(ortho_center(ψ))
  ψ[v] = f(ψ[v], x)
  return ψ
end

function _apply_to_orthocenter(f, ψ::AbstractTTN, x)
  return _apply_to_orthocenter!(f, copy(ψ), x)
end

Base.:*(ψ::AbstractTTN, α::Number) = _apply_to_orthocenter(*, ψ, α)

Base.:*(α::Number, ψ::AbstractTTN) = ψ * α

Base.:/(ψ::AbstractTTN, α::Number) = _apply_to_orthocenter(/, ψ, α)

Base.:-(ψ::AbstractTTN) = -1 * ψ

function LinearAlgebra.rmul!(ψ::AbstractTTN, α::Number)
  return _apply_to_orthocenter!(*, ψ, α)
end

function lognorm(ψ::AbstractTTN)
  if isortho(ψ)
    return log(norm(ψ[only(ortho_center(ψ))]))
  end
  lognorm2_ψ = loginner(ψ, ψ)
  rtol = eps(real(scalartype(ψ))) * 10
  atol = rtol
  if !IsApprox.isreal(lognorm2_ψ, Approx(; rtol=rtol, atol=atol))
    @warn "log(norm²) is $lognorm2_T, which is not real up to a relative tolerance of $rtol and an absolute tolerance of $atol. Taking the real part, which may not be accurate."
  end
  return 0.5 * real(lognorm2_ψ)
end

function logdot(ψ1::TTNT, ψ2::TTNT; kwargs...) where {TTNT<:AbstractTTN}
  return loginner(ψ1, ψ2; kwargs...)
end

#TODO: stick with this traversal or find optimal contraction sequence?
function loginner(
  ψ1::TTNT, ψ2::TTNT; root_vertex=default_root_vertex(ψ1, ψ2)
)::Number where {TTNT<:AbstractTTN}
  N = nv(ψ1)
  if nv(ψ2) != N
    throw(DimensionMismatch("inner: mismatched number of vertices $N and $(nv(ψ2))"))
  end
  ψ1dag = sim(dag(ψ1); sites=[])
  traversal_order = reverse(post_order_dfs_vertices(ψ1, root_vertex))

  O = ψ1dag[root_vertex] * ψ2[root_vertex]

  normO = norm(O)
  log_inner_tot = log(normO)
  O ./= normO

  for v in traversal_order[2:end]
    O = (O * ψ1dag[v]) * ψ2[v]
    normO = norm(O)
    log_inner_tot += log(normO)
    O ./= normO
  end

  if !isreal(O[]) || real(O[]) < 0
    log_inner_tot += log(complex(O[]))
  end
  return log_inner_tot
end

function _add_maxlinkdims(ψs::AbstractTTN...)
  maxdims = Dictionary{edgetype(ψs[1]),Int}()
  for e in edges(ψs[1])
    maxdims[e] = sum(ψ -> linkdim(ψ, e), ψs)
    maxdims[reverse(e)] = maxdims[e]
  end
  return maxdims
end

# TODO: actually implement this?
function Base.:+(
  ::ITensors.Algorithm"densitymatrix",
  ψs::TTNT...;
  cutoff=1e-15,
  root_vertex=default_root_vertex(ψs...),
  kwargs...,
) where {TTNT<:AbstractTTN}
  return error("Not implemented (yet) for trees.")
end

function Base.:+(
  ::ITensors.Algorithm"directsum", ψs::TTNT...; root_vertex=default_root_vertex(ψs...)
) where {TTNT<:AbstractTTN}
  @assert all(ψ -> nv(first(ψs)) == nv(ψ), ψs)

  # Output state
  ϕ = TTN(siteinds(ψs[1]))

  vs = post_order_dfs_vertices(ϕ, root_vertex)
  es = post_order_dfs_edges(ϕ, root_vertex)
  link_space = Dict{edgetype(ϕ),Index}()

  for v in reverse(vs)
    edges = filter(e -> dst(e) == v || src(e) == v, es)
    dims_in = findall(e -> dst(e) == v, edges)
    dim_out = findfirst(e -> src(e) == v, edges)

    ls = [Tuple(only(linkinds(ψ, e)) for e in edges) for ψ in ψs]
    ϕv, lv = directsum((ψs[i][v] => ls[i] for i in 1:length(ψs))...; tags=tags.(first(ls)))
    for din in dims_in
      link_space[edges[din]] = lv[din]
    end
    if !isnothing(dim_out)
      ϕv = replaceind(ϕv, lv[dim_out] => dag(link_space[edges[dim_out]]))
    end

    ϕ[v] = ϕv
  end
  return convert(TTNT, ϕ)
end

# TODO: switch default algorithm once more are implemented
function Base.:+(ψs::AbstractTTN...; alg=ITensors.Algorithm"directsum"(), kwargs...)
  return +(ITensors.Algorithm(alg), ψs...; kwargs...)
end

Base.:+(ψ::AbstractTTN) = ψ

ITensors.add(ψs::AbstractTTN...; kwargs...) = +(ψs...; kwargs...)

function Base.:-(ψ1::AbstractTTN, ψ2::AbstractTTN; kwargs...)
  return +(ψ1, -ψ2; kwargs...)
end

function ITensors.add(tn1::AbstractTTN, tn2::AbstractTTN; kwargs...)
  return +(tn1, tn2; kwargs...)
end

# TODO: Delete this
function permute(ψ::AbstractTTN, ::Tuple{typeof(linkind),typeof(siteinds),typeof(linkind)})
  ψ̃ = copy(ψ)
  for v in vertices(ψ)
    ls = [only(linkinds(ψ, n => v)) for n in neighbors(ψ, v)] # TODO: won't work for multiple indices per link...
    ss = TupleTools.sort(Tuple(siteinds(ψ, v)); by=plev)
    setindex_preserve_graph!(
      ψ̃, permute(ψ[v], filter(!isnothing, (ls[1], ss..., ls[2:end]...))), v
    )
  end
  ψ̃ = set_ortho_center(ψ̃, ortho_center(ψ))
  return ψ̃
end

function Base.isapprox(
  x::AbstractTTN,
  y::AbstractTTN;
  atol::Real=0,
  rtol::Real=Base.rtoldefault(
    LinearAlgebra.promote_leaf_eltypes(x), LinearAlgebra.promote_leaf_eltypes(y), atol
  ),
)
  d = norm(x - y)
  if isfinite(d)
    return d <= max(atol, rtol * max(norm(x), norm(y)))
  else
    error("In `isapprox(x::AbstractTTN, y::AbstractTTN)`, `norm(x - y)` is not finite")
  end
end

#
# Inner products
#

# TODO: implement using multi-graph disjoint union
function inner(
  y::AbstractTTN, A::AbstractTTN, x::AbstractTTN; root_vertex=default_root_vertex(x, A, y)
)
  traversal_order = reverse(post_order_dfs_vertices(x, root_vertex))
  ydag = sim(dag(y); sites=[])
  x = sim(x; sites=[])
  O = ydag[root_vertex] * A[root_vertex] * x[root_vertex]
  for v in traversal_order[2:end]
    O = O * ydag[v] * A[v] * x[v]
  end
  return O[]
end

# TODO: implement using multi-graph disjoint
function inner(
  B::AbstractTTN,
  y::AbstractTTN,
  A::AbstractTTN,
  x::AbstractTTN;
  root_vertex=default_root_vertex(B, y, A, x),
)
  N = nv(B)
  if nv(y) != N || nv(x) != N || nv(A) != N
    throw(
      DimensionMismatch(
        "inner: mismatched number of vertices $N and $(nv(x)) or $(nv(y)) or $(nv(A))"
      ),
    )
  end
  ydag = sim(linkinds, dag(y))
  Bdag = sim(linkinds, dag(B))
  traversal_order = reverse(post_order_dfs_vertices(x, root_vertex))
  yB = ydag[root_vertex] * Bdag[root_vertex]
  Ax = A[root_vertex] * x[root_vertex]
  O = yB * Ax
  for v in traversal_order[2:end]
    yB = ydag[v] * Bdag[v]
    Ax = A[v] * x[v]
    yB *= O
    O = yB * Ax
  end
  return O[]
end

function expect(
  operator::String,
  state::AbstractTTN;
  vertices=vertices(state),
  # ToDo: verify that this is a sane default
  root_vertex=default_root_vertex(siteinds(state)),
)
  # ToDo: for performance it may be beneficial to also implement expect! and change the orthogonality center in place
  # assuming that the next algorithmic step can make use of the orthogonality center being moved to a different vertex
  # ToDo: Verify that this is indeed the correct order for performance
  sites = siteinds(state)
  ordered_vertices = reverse(post_order_dfs_vertices(sites, root_vertex))
  res = Dictionary(vertices, undef)
  for v in ordered_vertices
    !(v in vertices) && continue
    state = orthogonalize(state, v)
    @assert isone(length(sites[v]))
    #ToDo: Add compatibility with more than a single index per vertex
    op_v = op(operator, only(sites[v]))
    res[v] = (dag(state[v]) * apply(op_v, state[v]))[]
  end
  return mapreduce(typeof, promote_type, res).(res)
end
