default_inner_partitioned_vertices(tn) = group(v -> first(v), vertices(tn))
#Default to BP always?!
default_algorithm(tns::Vector) = all(is_tree.(tns)) ? "bp" : "exact"

#Default for map_linkinds should be sim.
#Use form code and just default to identity inbetween x and y
#Have ϕ in the same space as y and then a dual_map kwarg?

function inner(
  ϕ::AbstractITensorNetwork,
  ψ::AbstractITensorNetwork;
  alg=default_algorithm([ϕ, ψ]),
  kwargs...,
)
  return inner(Algorithm(alg), ϕ, ψ; kwargs...)
end

#Make [A, ϕ, ψ] a Tuple
function inner(
  ϕ::AbstractITensorNetwork,
  A::AbstractITensorNetwork,
  ψ::AbstractITensorNetwork;
  alg=default_algorithm([ϕ, A, ψ]),
  kwargs...,
)
  return inner(Algorithm(alg), ϕ, A, ψ; kwargs...)
end

function inner(
  alg::Algorithm"exact",
  ϕ::AbstractITensorNetwork,
  ψ::AbstractITensorNetwork;
  sequence=nothing,
  contraction_sequence_kwargs=(;),
  kwargs...,
)
  tn = inner_network(ϕ, ψ; kwargs...)
  if isnothing(sequence)
    sequence = contraction_sequence(tn; contraction_sequence_kwargs...)
  end
  return scalar(tn; sequence)
end

function inner(
  alg::Algorithm"exact",
  ϕ::AbstractITensorNetwork,
  A::AbstractITensorNetwork,
  ψ::AbstractITensorNetwork;
  sequence=nothing,
  contraction_sequence_kwargs=(;),
  site_index_map=prime,
  kwargs...,
)
  tn = inner_network(ϕ, A, ψ; kwargs...)
  if isnothing(sequence)
    sequence = contraction_sequence(tn; contraction_sequence_kwargs...)
  end
  return scalar(tn; sequence)
end

function loginner(
  ϕ::AbstractITensorNetwork,
  ψ::AbstractITensorNetwork;
  alg=default_algorithm(ϕ, ψ),
  kwargs...,
)
  return loginner(Algorithm(alg), ϕ, ψ; kwargs...)
end

function loginner(
  ϕ::AbstractITensorNetwork,
  A::AbstractITensorNetwork,
  ψ::AbstractITensorNetwork;
  alg=default_algorithm(ϕ, ψ),
  kwargs...,
)
  return loginner(Algorithm(alg), A, ϕ, ψ; kwargs...)
end

function loginner(
  alg::Algorithm"exact", ϕ::AbstractITensorNetwork, ψ::AbstractITensorNetwork; kwargs...
)
  return log(inner(alg, ϕ, ψ); kwargs...)
end

function loginner(
  alg::Algorithm"exact",
  ϕ::AbstractITensorNetwork,
  A::AbstractITensorNetwork,
  ψ::AbstractITensorNetwork;
  kwargs...,
)
  return log(inner(alg, ϕ, A, ψ); kwargs...)
end

function loginner(
  alg::Algorithm"bp",
  ϕ::AbstractITensorNetwork,
  ψ::AbstractITensorNetwork;
  partitioned_verts=default_inner_partitioned_vertices,
  dual_link_index_map=sim,
  kwargs...,
)
  tn = inner_network(ϕ, ψ; dual_link_index_map)
  return logscalar(alg, tn; partitioned_vertices=partitioned_verts(tn), kwargs...)
end

function loginner(
  alg::Algorithm"bp",
  ϕ::AbstractITensorNetwork,
  A::AbstractITensorNetwork,
  ψ::AbstractITensorNetwork;
  partitioned_verts=default_inner_partitioned_vertices,
  dual_link_index_map=sim,
  kwargs...,
)
  tn = inner_network(ϕ, A, ψ; dual_link_index_map)
  return logscalar(alg, tn; partitioned_vertices=partitioned_verts(tn), kwargs...)
end

function inner(
  alg::Algorithm"bp",
  ϕ::AbstractITensorNetwork,
  ψ::AbstractITensorNetwork;
  partitioned_verts=default_inner_partitioned_vertices,
  dual_link_index_map=prime,
  kwargs...,
)
  tn = inner_network(ϕ, ψ; dual_link_index_map)
  return scalar(alg, tn; partitioned_vertices=partitioned_verts(tn), kwargs...)
end

function inner(
  alg::Algorithm"bp",
  A::AbstractITensorNetwork,
  ϕ::AbstractITensorNetwork,
  ψ::AbstractITensorNetwork;
  partitioned_verts=default_inner_partitioned_vertices,
  dual_link_index_map=prime,
  kwargs...,
)
  tn = inner_network(ϕ, A, ψ; dual_link_index_map)
  return scalar(alg, tn; partitioned_vertices=partitioned_verts(tn), kwargs...)
end

# TODO: rename `sqnorm` to match https://github.com/JuliaStats/Distances.jl,
# or `norm_sqr` to match `LinearAlgebra.norm_sqr`
norm_sqr(ψ::AbstractITensorNetwork; kwargs...) = inner(ψ, ψ; kwargs...)

function norm(ψ::AbstractITensorNetwork; kwargs...)
  return sqrt(abs(real(norm_sqr(ψ; kwargs...))))
end
