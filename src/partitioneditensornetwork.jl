function linkinds(pitn::PartitionedGraph, edge::PartitionEdge)
  src_e_itn = subgraph(pitn, src(edge))
  dst_e_itn = subgraph(pitn, dst(edge))
  return commoninds(src_e_itn, dst_e_itn)
end
