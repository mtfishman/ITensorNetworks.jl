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
)
  tn = disjoint_union(
    operator_vertex_suffix => operator, bra_vertex_suffix => bra, ket_vertex_suffix => ket
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

function Base.copy(blf::BilinearFormNetwork)
  return BilinearFormNetwork(
    copy(tensornetwork(blf)),
    operator_vertex_suffix(blf),
    bra_vertex_suffix(blf),
    ket_vertex_suffix(blf),
  )
end

function BilinearFormNetwork(
  bra::AbstractITensorNetwork, ket::AbstractITensorNetwork; kwargs...
)
  operator_inds = union_all_inds(siteinds(bra), siteinds(ket))
  O = delta_network(operator_inds)
  return BilinearFormNetwork(O, bra, ket; kwargs...)
end

function update(
  blf::BilinearFormNetwork, original_state_vertex, bra_state::ITensor, ket_state::ITensor
)
  blf = copy(blf)
  # TODO: Maybe add a check that it really does preserve the graph.
  setindex_preserve_graph!(
    tensornetwork(blf), bra_state, bra_vertex(blf, original_state_vertex)
  )
  setindex_preserve_graph!(
    tensornetwork(blf), ket_state, ket_vertex(blf, original_state_vertex)
  )
  return blf
end
