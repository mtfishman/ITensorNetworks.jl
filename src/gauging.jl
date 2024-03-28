function default_bond_tensors(ψ::ITensorNetwork)
  return DataGraph{vertextype(ψ),Nothing,ITensor}(underlying_graph(ψ))
end

struct VidalITensorNetwork{V,BTS} <: AbstractITensorNetwork{V}
  itensornetwork::ITensorNetwork{V}
  bond_tensors::BTS
end

site_tensors(ψ::VidalITensorNetwork) = ψ.itensornetwork
bond_tensors(ψ::VidalITensorNetwork) = ψ.bond_tensors
bond_tensor(ψ::VidalITensorNetwork, e) = bond_tensors(ψ)[e]

function data_graph_type(TN::Type{<:VidalITensorNetwork})
  return data_graph_type(fieldtype(TN, :itensornetwork))
end
data_graph(ψ::VidalITensorNetwork) = data_graph(site_tensors(ψ))
function copy(ψ::VidalITensorNetwork)
  return VidalITensorNetwork(copy(site_tensors(ψ)), copy(bond_tensors(ψ)))
end

function default_norm_cache(ψ::ITensorNetwork)
  ψψ = norm_sqr_network(ψ)
  return BeliefPropagationCache(ψψ, group(v -> v[1], vertices(ψψ)))
end

function ITensorNetwork(
  ψ_vidal::VidalITensorNetwork; (cache!)=nothing, update_gauge=false, update_kwargs...
)
  if update_gauge
    ψ_vidal = update(ψ_vidal; update_kwargs...)
  end

  ψ = copy(site_tensors(ψ_vidal))

  for e in edges(ψ)
    vsrc, vdst = src(e), dst(e)
    root_S = sqrt_diag(bond_tensor(ψ_vidal, e))
    setindex_preserve_graph!(ψ, noprime(root_S * ψ[vsrc]), vsrc)
    setindex_preserve_graph!(ψ, noprime(root_S * ψ[vdst]), vdst)
  end

  if !isnothing(cache!)
    bp_cache = default_norm_cache(ψ)
    mts = messages(bp_cache)

    for e in edges(ψ)
      vsrc, vdst = src(e), dst(e)
      pe = partitionedge(bp_cache, (vsrc, 1) => (vdst, 1))
      set!(mts, pe, copy(ITensor[dense(bond_tensor(ψ_vidal, e))]))
      set!(mts, reverse(pe), copy(ITensor[dense(bond_tensor(ψ_vidal, e))]))
    end

    bp_cache = set_messages(bp_cache, mts)
    cache![] = bp_cache
  end

  return ψ
end

"""Use an ITensorNetwork ψ, its bond tensors and belief propagation cache to put ψ into the vidal gauge, return the bond tensors and updated_ψ."""
function vidalitensornetwork_preserve_cache(
  ψ::ITensorNetwork;
  cache=default_norm_cache(ψ),
  bond_tensors=default_bond_tensors,
  message_cutoff=10 * eps(real(scalartype(ψ))),
  regularization=10 * eps(real(scalartype(ψ))),
  edges=NamedGraphs.edges(ψ),
  svd_kwargs...,
)
  ψ_vidal_site_tensors = copy(ψ)
  ψ_vidal_bond_tensors = bond_tensors(ψ)

  for e in edges
    vsrc, vdst = src(e), dst(e)
    ψvsrc, ψvdst = ψ_vidal_site_tensors[vsrc], ψ_vidal_site_tensors[vdst]

    pe = partitionedge(cache, (vsrc, 1) => (vdst, 1))
    edge_ind = commoninds(ψvsrc, ψvdst)
    edge_ind_sim = sim(edge_ind)

    X_D, X_U = eigen(only(message(cache, pe)); ishermitian=true, cutoff=message_cutoff)
    Y_D, Y_U = eigen(
      only(message(cache, reverse(pe))); ishermitian=true, cutoff=message_cutoff
    )
    X_D, Y_D = map_diag(x -> x + regularization, X_D),
    map_diag(x -> x + regularization, Y_D)

    rootX_D, rootY_D = sqrt_diag(X_D), sqrt_diag(Y_D)
    inv_rootX_D, inv_rootY_D = invsqrt_diag(X_D), invsqrt_diag(Y_D)
    rootX = X_U * rootX_D * prime(dag(X_U))
    rootY = Y_U * rootY_D * prime(dag(Y_U))
    inv_rootX = X_U * inv_rootX_D * prime(dag(X_U))
    inv_rootY = Y_U * inv_rootY_D * prime(dag(Y_U))

    ψvsrc, ψvdst = noprime(ψvsrc * inv_rootX), noprime(ψvdst * inv_rootY)

    Ce = rootX
    Ce = Ce * replaceinds(rootY, edge_ind, edge_ind_sim)

    U, S, V = svd(Ce, edge_ind; svd_kwargs...)

    new_edge_ind = Index[Index(dim(commoninds(S, U)), tags(first(edge_ind)))]

    ψvsrc = replaceinds(ψvsrc * U, commoninds(S, U), new_edge_ind)
    ψvdst = replaceinds(ψvdst, edge_ind, edge_ind_sim)
    ψvdst = replaceinds(ψvdst * V, commoninds(V, S), new_edge_ind)

    setindex_preserve_graph!(ψ_vidal_site_tensors, ψvsrc, vsrc)
    setindex_preserve_graph!(ψ_vidal_site_tensors, ψvdst, vdst)

    S = replaceinds(
      S,
      [commoninds(S, U)..., commoninds(S, V)...] =>
        [new_edge_ind..., prime(new_edge_ind)...],
    )
    ψ_vidal_bond_tensors[e] = S
  end

  return VidalITensorNetwork(ψ_vidal_site_tensors, ψ_vidal_bond_tensors)
end

function VidalITensorNetwork(
  ψ::ITensorNetwork;
  (cache!)=nothing,
  update_cache=isnothing(cache!),
  cache_update_kwargs=default_cache_update_kwargs(cache!),
  kwargs...,
)
  if isnothing(cache!)
    cache! = Ref(default_norm_cache(ψ))
  end
  cache![] = update(cache![]; cache_update_kwargs...)
  return vidalitensornetwork_preserve_cache(ψ; cache=cache![], kwargs...)
end

function update(ψ::VidalITensorNetwork; kwargs...)
  return VidalITensorNetwork(ITensorNetwork(ψ; update_gauge=false); kwargs...)
end

"""Function to construct the 'isometry' of a state in the Vidal Gauge on the given edge"""
function vidal_gauge_isometry(ψ::VidalITensorNetwork, edge)
  vsrc, vdst = src(edge), dst(edge)
  ψ_vsrc = copy(ψ[vsrc])

  for vn in setdiff(neighbors(ψ, vsrc), [vdst])
    ψ_vsrc = noprime(ψ_vsrc * bond_tensor(ψ, vn => vsrc))
  end

  ψ_vsrcdag = dag(ψ_vsrc)
  ψ_vsrcdag = replaceind(ψ_vsrcdag, commonind(ψ_vsrc, ψ[vdst]), commonind(ψ_vsrc, ψ[vdst])')

  return ψ_vsrcdag * ψ_vsrc
end

function vidal_gauge_isometries(ψ::VidalITensorNetwork, edges::Vector)
  return Dict([e => vidal_gauge_isometry(ψ, e) for e in edges])
end

function vidal_gauge_isometries(ψ::VidalITensorNetwork)
  return vidal_gauge_isometries(
    ψ, vcat(NamedGraphs.edges(ψ), reverse.(NamedGraphs.edges(ψ)))
  )
end

"""Function to measure the 'distance' of a state from the Vidal Gauge"""
function gauge_error(ψ::VidalITensorNetwork)
  f = 0
  isometries = vidal_gauge_isometries(ψ)
  for e in keys(isometries)
    lhs = isometries[e]
    f += message_diff(ITensor[lhs], ITensor[denseblocks(delta(inds(lhs)))])
  end

  return f / (length(keys(isometries)))
end
