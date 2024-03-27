using NamedGraphs: incident_edges

struct ProjOuterProdTTN{V} <: AbstractProjTTN{V}
  pos::Union{Vector{<:V},NamedEdge{V}}
  internal_state::TTN{V}
  operator::TTN{V}
  environments::Dictionary{NamedEdge{V},ITensor}
end

environments(p::ProjOuterProdTTN) = p.environments
operator(p::ProjOuterProdTTN) = p.operator
underlying_graph(p::ProjOuterProdTTN) = underlying_graph(operator(p))
pos(p::ProjOuterProdTTN) = p.pos
internal_state(p::ProjOuterProdTTN) = p.internal_state

function ProjOuterProdTTN(internal_state::AbstractTTN, operator::AbstractTTN)
  return ProjOuterProdTTN(
    vertextype(operator)[],
    internal_state,
    operator,
    Dictionary{edgetype(operator),ITensor}(),
  )
end

function Base.copy(P::ProjOuterProdTTN)
  return ProjOuterProdTTN(
    pos(P), copy(internal_state(P)), copy(operator(P)), copy(environments(P))
  )
end

function set_nsite(P::ProjOuterProdTTN, nsite)
  return P
end

function shift_position(P::ProjOuterProdTTN, pos)
  return ProjOuterProdTTN(pos, internal_state(P), operator(P), environments(P))
end

function set_environments(p::ProjOuterProdTTN, environments)
  return ProjOuterProdTTN(pos(p), internal_state(p), operator(p), environments)
end

set_environment(p::ProjOuterProdTTN, edge, env) = set_environment!(copy(p), edge, env)
function set_environment!(p::ProjOuterProdTTN, edge, env)
  set!(environments(p), edge, env)
  return p
end

function make_environment(P::ProjOuterProdTTN, state::AbstractTTN, e::AbstractEdge)
  # invalidate environment for opposite edge direction if necessary
  reverse(e) ∈ incident_edges(P) || (P = invalidate_environment(P, reverse(e)))
  # do nothing if valid environment already present
  if !haskey(environments(P), e)
    if is_leaf(underlying_graph(P), src(e))
      # leaves are easy
      env = internal_state(P)[src(e)] * operator(P)[src(e)] * dag(state[src(e)])
    else
      # construct by contracting neighbors
      neighbor_envs = ITensor[]
      for n in setdiff(neighbors(underlying_graph(P), src(e)), [dst(e)])
        P = make_environment(P, state, edgetype(P)(n, src(e)))
        push!(neighbor_envs, environment(P, edgetype(P)(n, src(e))))
      end
      # manually heuristic for contraction order: two environments, site tensors, then
      # other environments
      frst, scnd, rst = _separate_first_two(neighbor_envs)
      itensor_map = vcat(
        internal_state(P)[src(e)], frst, scnd, operator(P)[src(e)], dag(state[src(e)]), rst
      ) # no prime here in comparison to the same routine for Projttn
      # TODO: actually use optimal contraction sequence here
      env = reduce(*, itensor_map)
    end
    P = set_environment(P, e, env)
  end
  @assert(
    hascommoninds(environment(P, e), state[src(e)]),
    "Something went wrong, probably re-orthogonalized this edge in the same direction twice!"
  )
  return P
end

function projected_operator_tensors(P::ProjOuterProdTTN)
  environments = ITensor[environment(P, edge) for edge in incident_edges(P)]
  # manual heuristic for contraction order fixing: for each site in ProjTTN, apply up to
  # two environments, then TTN tensor, then other environments
  itensor_map = Union{ITensor,OneITensor}[] # TODO: will a Hamiltonian TTN tensor ever be a OneITensor?
  for j in sites(P)
    push!(itensor_map, internal_state(P)[j])
  end
  if on_edge(P)
    append!(itensor_map, environments)
  else
    for s in sites(P)
      site_envs = filter(hascommoninds(operator(P)[s]), environments)
      frst, scnd, rst = _separate_first_two(site_envs)
      site_tensors = vcat(frst, scnd, operator(P)[s], rst)
      append!(itensor_map, site_tensors)
    end
  end
  return itensor_map
end

function contract_ket(P::ProjOuterProdTTN, v::ITensor)
  itensor_map = projected_operator_tensors(P)
  for t in itensor_map
    v *= t
  end
  return v
end

# ToDo: verify conjugation etc. with complex AbstractTTN
function contract(P::ProjOuterProdTTN, x::ITensor)
  ket = contract_ket(P, ITensor(one(Bool)))
  return (dag(ket) * x) * ket
end
