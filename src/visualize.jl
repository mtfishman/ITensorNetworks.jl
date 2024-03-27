using DataGraphs: AbstractDataGraph, underlying_graph
using Graphs: vertices
using ITensors.ITensorVisualizationCore: ITensorVisualizationCore, visualize
using NamedGraphs: AbstractNamedGraph, parent_graph

function ITensorVisualizationCore.visualize(
  graph::AbstractNamedGraph,
  args...;
  vertex_labels_prefix=nothing,
  vertex_labels=nothing,
  kwargs...,
)
  if !isnothing(vertex_labels_prefix)
    vertex_labels = [vertex_labels_prefix * string(v) for v in vertices(graph)]
  end
  #edge_labels = [string(e) for e in edges(graph)]
  return visualize(parent_graph(graph), args...; vertex_labels, kwargs...)
end

function ITensorVisualizationCore.visualize(graph::AbstractDataGraph, args...; kwargs...)
  return visualize(underlying_graph(graph), args...; kwargs...)
end
