default_dual_site_index_map = prime
default_dual_link_index_map = sim

struct BilinearFormNetwork{
  V,
  TensorNetwork<:AbstractITensorNetwork{V},
  OperatorVertexSuffix,
  BraVertexSuffix,
  KetVertexSuffix,
} <: AbstractFormNetwork{V}
  tensornetwork::TensorNetwork
  operator_vertex_suffix::OperatorVertexSuffix
  bra_vertex_suffix::BraVertexSuffix
  ket_vertex_suffix::KetVertexSuffix
end

function BilinearFormNetwork(
  operator::AbstractITensorNetwork,
  bra::AbstractITensorNetwork,
  ket::AbstractITensorNetwork;
  operator_vertex_suffix=default_operator_vertex_suffix(),
  bra_vertex_suffix=default_bra_vertex_suffix(),
  ket_vertex_suffix=default_ket_vertex_suffix(),
  dual_site_index_map = default_dual_site_index_map,
  dual_link_index_map = default_dual_link_index_map
)
  bra_mapped = dual_link_index_map(dual_site_index_map(bra; links = []); sites = []) 
  tn = disjoint_union(
    operator_vertex_suffix => operator, bra_vertex_suffix => dag(bra_mapped), ket_vertex_suffix => ket
  )
  return BilinearFormNetwork(
    tn, operator_vertex_suffix, bra_vertex_suffix, ket_vertex_suffix
  )
end

operator_vertex_suffix(blf::BilinearFormNetwork) = blf.operator_vertex_suffix
bra_vertex_suffix(blf::BilinearFormNetwork) = blf.bra_vertex_suffix
ket_vertex_suffix(blf::BilinearFormNetwork) = blf.ket_vertex_suffix
tensornetwork(blf::BilinearFormNetwork) = blf.tensornetwork
data_graph_type(::Type{<:BilinearFormNetwork}) = data_graph_type(tensornetwork(blf))
data_graph(blf::BilinearFormNetwork) = data_graph(tensornetwork(blf))

function copy(blf::BilinearFormNetwork)
  return BilinearFormNetwork(
    copy(tensornetwork(blf)),
    operator_vertex_suffix(blf),
    bra_vertex_suffix(blf),
    ket_vertex_suffix(blf),
  )
end

#Is the ordering of the indices correct here? CHECK THIS
#Put bra into the vector space!!!!
function BilinearFormNetwork(
  bra::AbstractITensorNetwork, ket::AbstractITensorNetwork; 
  dual_site_index_map = default_dual_site_index_map,
  kwargs...
)
  @assert issetequal(externalinds(bra), externalinds(ket))
  operator_inds = union_all_inds(siteinds(ket), dual_site_index_map(siteinds(ket)))
  O = delta_network(operator_inds)
  return BilinearFormNetwork(O, bra, ket; dual_site_index_map, kwargs...)
end

function update(
  blf::BilinearFormNetwork, original_bra_state_vertex, original_ket_state_vertex, bra_state::ITensor, ket_state::ITensor
)
  blf = copy(blf)
  # TODO: Maybe add a check that it really does preserve the graph.
  setindex_preserve_graph!(
    tensornetwork(blf), bra_state, bra_vertex(blf, original_bra_state_vertex)
  )
  setindex_preserve_graph!(
    tensornetwork(blf), ket_state, ket_vertex(blf, original_ket_state_vertex)
  )
  return blf
end
