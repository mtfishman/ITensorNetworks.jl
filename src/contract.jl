function contract(tn::AbstractITensorNetwork; alg::String="exact", kwargs...)
  return contract(Algorithm(alg), tn; kwargs...)
end

function contract(
  alg::Algorithm"exact", tn::AbstractITensorNetwork; sequence=vertices(tn), kwargs...
)
  sequence_linear_index = deepmap(v -> vertex_to_parent_vertex(tn, v), sequence)
  return contract(Vector{ITensor}(tn); sequence=sequence_linear_index, kwargs...)
end

contract(alg::Algorithm"exact", tn::Vector{ITensor}; kwargs...) = contract(tn; kwargs...)

function contract(
  alg::Union{Algorithm"density_matrix",Algorithm"ttn_svd"},
  tn::AbstractITensorNetwork;
  output_structure::Function=path_graph_structure,
  kwargs...,
)
  return approx_itensornetwork(alg, tn, output_structure; kwargs...)
end

function contract_density_matrix(
  contract_list::Vector{ITensor}; normalize=true, contractor_kwargs...
)
  tn, _ = contract(
    ITensorNetwork(contract_list); alg="density_matrix", contractor_kwargs...
  )
  out = Vector{ITensor}(tn)
  if normalize
    out .= normalize!.(copy.(out))
  end
  return out
end

function scalar(
  alg::Algorithm, tn::Union{AbstractITensorNetwork,Vector{ITensor}}; kwargs...
)
  return contract(alg, tn; kwargs...)[]
end

function scalar(tn::Union{AbstractITensorNetwork,Vector{ITensor}}; alg="exact", kwargs...)
  return scalar(Algorithm(alg), tn; kwargs...)
end

function logscalar(
  tn::Union{AbstractITensorNetwork,Vector{ITensor}}; alg::String="exact", kwargs...
)
  return logscalar(Algorithm(alg), tn; kwargs...)
end

function logscalar(
  alg::Algorithm"exact", tn::Union{AbstractITensorNetwork,Vector{ITensor}}; kwargs...
)
  return log(complex(scalar(alg, tn; kwargs...)))
end

#This should just pass to a logscalar(bp_cache::BeliefPropagationCache, ...)
#Catch the 0 case in logscalar not in scalar, then pass to exp(logscalar(...))
#Check if negative before complex() call
#Break down into scalar factors? 
#Make general to all algorithms

function logscalar(
  alg::Algorithm"bp",
  tn::AbstractITensorNetwork;
  (cache!)=nothing,
  partitioned_vertices=default_partitioned_vertices(tn),
  update_cache=isnothing(cache!),
  cache_update_kwargs=default_cache_update_kwargs(cache!),
)
  if isnothing(cache!)
    cache! = Ref(BeliefPropagationCache(tn, partitioned_vertices))
  end

  if update_cache
    cache![] = update(cache![]; cache_update_kwargs...)
  end

  numerator_terms, denominator_terms = vertex_norms(cache![]), edge_norms(cache![])

  return sum(log.(complex.(numerator_terms))) - sum(log.(complex.((denominator_terms))))
end

function scalar(
  alg::Algorithm"bp",
  tn::AbstractITensorNetwork;
  (cache!)=nothing,
  partitioned_vertices=default_partitioned_vertices(tn),
  update_cache=isnothing(cache!),
  cache_update_kwargs=default_cache_update_kwargs(cache!),
)
  if isnothing(cache!)
    cache! = Ref(BeliefPropagationCache(tn, partitioned_vertices))
  end

  if update_cache
    cache![] = update(cache![]; cache_update_kwargs...)
  end

  numerator_terms, denominator_terms = vertex_norms(cache![]), edge_norms(cache![])

  return prod(numerator_terms) / prod(denominator_terms)
end
