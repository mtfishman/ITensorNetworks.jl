using DataGraphs: DataGraphs, underlying_graph
using Dictionaries: Dictionary
using Graphs: edgetype, vertices
using ITensors: ITensor
using NamedGraphs: NamedEdge, incident_edges

"""
ProjTTN
"""
struct ProjTTN{V} <: AbstractProjTTN{V}
  pos::Union{Vector{<:V},NamedEdge{V}} # TODO: cleanest way to specify effective Hamiltonian position?
  operator::TTN{V}
  environments::Dictionary{NamedEdge{V},ITensor}
end

function ProjTTN(operator::TTN)
  return ProjTTN(vertices(operator), operator, Dictionary{edgetype(operator),ITensor}())
end

Base.copy(P::ProjTTN) = ProjTTN(pos(P), copy(operator(P)), copy(environments(P)))

#accessors for fields
environments(p::ProjTTN) = p.environments
operator(p::ProjTTN) = p.operator
DataGraphs.underlying_graph(P::ProjTTN) = underlying_graph(operator(P))
pos(P::ProjTTN) = P.pos

# trivial if we choose to specify position as above; only kept to allow using alongside
# ProjMPO
function set_nsite(P::ProjTTN, nsite)
  return P
end

function shift_position(P::ProjTTN, pos)
  return ProjTTN(pos, operator(P), environments(P))
end

set_environments(p::ProjTTN, environments) = ProjTTN(pos(p), operator(p), environments)
set_environment(p::ProjTTN, edge, env) = set_environment!(copy(p), edge, env)
function set_environment!(p::ProjTTN, edge, env)
  set!(environments(p), edge, env)
  return p
end

function make_environment(P::ProjTTN, state::AbstractTTN, e::AbstractEdge)
  # invalidate environment for opposite edge direction if necessary
  reverse(e) ∈ incident_edges(P) || (P = invalidate_environment(P, reverse(e)))
  # do nothing if valid environment already present
  if !haskey(environments(P), e)
    if is_leaf(underlying_graph(P), src(e))
      # leaves are easy
      env = state[src(e)] * operator(P)[src(e)] * dag(prime(state[src(e)]))
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
        state[src(e)], frst, scnd, operator(P)[src(e)], dag(prime(state[src(e)])), rst
      )
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

function projected_operator_tensors(P::ProjTTN)
  environments = ITensor[environment(P, edge) for edge in incident_edges(P)]
  # manual heuristic for contraction order fixing: for each site in ProjTTN, apply up to
  # two environments, then TTN tensor, then other environments
  if on_edge(P)
    itensor_map = environments
  else
    itensor_map = Union{ITensor,OneITensor}[] # TODO: will a Hamiltonian TTN tensor ever be a OneITensor?
    for s in sites(P)
      site_envs = filter(hascommoninds(operator(P)[s]), environments)
      frst, scnd, rst = _separate_first_two(site_envs)
      site_tensors = vcat(frst, scnd, operator(P)[s], rst)
      append!(itensor_map, site_tensors)
    end
  end
  return itensor_map
end
