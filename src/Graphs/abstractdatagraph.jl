using DataGraphs: DataGraphs, AbstractDataGraph, underlying_graph
using NamedGraphs: AbstractNamedGraph

# TODO: we may want to move these to `DataGraphs.jl`
for f in [:_root, :_is_rooted, :_is_rooted_directed_binary_tree]
  @eval begin
    function $f(graph::AbstractDataGraph, args...; kwargs...)
      return $f(underlying_graph(graph), args...; kwargs...)
    end
  end
end

DataGraphs.edge_data_type(::AbstractNamedGraph) = Any

Base.isassigned(::AbstractNamedGraph, ::Any) = false

function Base.iterate(::AbstractDataGraph)
  return error(
    "Iterating data graphs is not yet defined. We may define it in the future as iterating through the vertex and edge data.",
  )
end
