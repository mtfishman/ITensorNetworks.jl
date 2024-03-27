default_index_map = prime
default_inv_index_map = noprime

struct QuadraticFormNetwork{V,FormNetwork<:BilinearFormNetwork{V},IndexMap,InvIndexMap} <:
       AbstractFormNetwork{V}
  formnetwork::FormNetwork
  dual_index_map::IndexMap
  dual_inv_index_map::InvIndexMap
end

bilinear_formnetwork(qf::QuadraticFormNetwork) = qf.formnetwork

#Needed for implementation, forward from bilinear form
for f in [
  :operator_vertex_suffix,
  :bra_vertex_suffix,
  :ket_vertex_suffix,
  :tensornetwork,
  :data_graph,
  :data_graph_type,
]
  @eval begin
    function $f(qf::QuadraticFormNetwork, args...; kwargs...)
      return $f(bilinear_formnetwork(qf), args...; kwargs...)
    end
  end
end

dual_index_map(qf::QuadraticFormNetwork) = qf.dual_index_map
dual_inv_index_map(qf::QuadraticFormNetwork) = qf.dual_inv_index_map
function Base.copy(qf::QuadraticFormNetwork)
  return QuadraticFormNetwork(
    copy(bilinear_formnetwork(qf)), dual_index_map(qf), dual_inv_index_map(qf)
  )
end

function QuadraticFormNetwork(
  operator::AbstractITensorNetwork,
  ket::AbstractITensorNetwork;
  dual_index_map=default_index_map,
  dual_inv_index_map=default_inv_index_map,
  kwargs...,
)
  bra = map_inds(dual_index_map, dag(ket))
  blf = BilinearFormNetwork(operator, bra, ket; kwargs...)
  return QuadraticFormNetwork(blf, dual_index_map, dual_inv_index_map)
end

function QuadraticFormNetwork(
  ket::AbstractITensorNetwork;
  dual_index_map=default_index_map,
  dual_inv_index_map=default_inv_index_map,
  kwargs...,
)
  bra = map_inds(dual_index_map, dag(ket))
  blf = BilinearFormNetwork(bra, ket; kwargs...)
  return QuadraticFormNetwork(blf, dual_index_map, dual_inv_index_map)
end

function update(qf::QuadraticFormNetwork, original_state_vertex, ket_state::ITensor)
  state_inds = inds(ket_state)
  bra_state = replaceinds(dag(ket_state), state_inds, dual_index_map(qf).(state_inds))
  new_blf = update(bilinear_formnetwork(qf), original_state_vertex, bra_state, ket_state)
  return QuadraticFormNetwork(new_blf, dual_index_map(qf), dual_index_map(qf))
end
