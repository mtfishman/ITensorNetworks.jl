function prepare_rsvd(A::Vector{ITensor}, linds::Vector{<:Index})
  if length(A) == 1
    rinds = uniqueinds(only(A), linds)
  else
    rinds = uniqueinds(A[end], unioninds(A[1:(end - 1)]...))
  end
  CL = combiner(linds)
  CR = combiner(rinds)
  AC = copy(A)
  AC[1] = CL * first(AC)
  AC[end] = last(AC) * CR
  cL = combinedind(CL)
  cR = combinedind(CR)
  return AC, (CL, cL), (CR, cR)
end

function rsvd_iterative(A::ITensor, linds::Vector{<:Index}; svd_kwargs...)
  return rsvd_iterative([A], linds; svd_kwargs...)
end

# ToDo: not type stable for empty factorizations.
function rsvd_iterative(A::Vector{ITensor}, linds::Vector{<:Index}; svd_kwargs...)
  AC, (CL, cL), (CR, cR) = prepare_rsvd(A, linds)
  scalar_type = eltype(first(AC))
  nonzero_sectors = get_column_space(AC, cL, cR)
  isempty(nonzero_sectors) && return nothing, nothing, nothing

  n_init = 1
  p_rule(n) = 2 * n
  ndict = init_guess_sizes(cR, nonzero_sectors, n_init, p_rule)

  M = build_guess_matrix(scalar_type, cR, nonzero_sectors, ndict)
  fact, Q = rsvd_core(AC, M; svd_kwargs...)
  n_inc = 1
  ndict = increment_guess_sizes(ndict, n_inc, p_rule)
  new_fact = nothing
  while true
    M = build_guess_matrix(scalar_type, cR, nonzero_sectors, ndict)
    new_fact, Q = rsvd_core(AC, M; svd_kwargs...)
    isnothing(Q) && return nothing, nothing, nothing
    if is_converged!(ndict, fact, new_fact; n_inc, has_qns=any(hasqns.(AC)), svd_kwargs...)
      break
    else
      fact = new_fact
    end
  end
  vals = diag(array(new_fact.S))
  (length(vals) == 1 && vals[1]^2 ≤ get(svd_kwargs, :cutoff, 0.0)) &&
    return nothing, nothing, nothing
  return dag(CL) * Q * new_fact.U, new_fact.S, new_fact.V * dag(CR)
end

function rsvd(A::ITensor, linds::Vector{<:Index}, n::Int, p::Int; svd_kwargs...)
  return rsvd([A], linds, n, p; svd_kwargs...)
end

function rsvd(A::Vector{<:ITensor}, linds::Vector{<:Index}, n::Int, p::Int; svd_kwargs...)
  AC, (CL, cL), (CR, cR) = prepare_rsvd(A, linds)
  nonzero_sectors = get_column_space(AC, cL, cR)
  isempty(nonzero_sectors) && return nothing, nothing, nothing
  M = build_guess_matrix(eltype(first(AC)), cR, nonzero_sectors, n, p)
  fact, Q = rsvd_core(AC, M; svd_kwargs...)
  return dag(CL) * Q * fact.U, fact.S, fact.V * dag(CR)
end

function rsvd_core(AC::Vector{ITensor}, M; svd_kwargs...)
  @assert !isnothing(commonind(last(AC), M))
  Q = foldr(*, AC; init=M)
  Q = ITensors.qr(Q, commoninds(Q, first(AC)))[1]
  any(isequal(0), dims(Q)) && return nothing, nothing, nothing, nothing
  QAC = foldl(*, AC; init=dag(Q))
  @assert typeof(QAC) <: ITensor
  fact = svd(QAC, commoninds(dag(Q), QAC); svd_kwargs...)
  return fact, Q
end

# ToDo : Remove this, not needed since everything passes via Vector of ITensors.
#=
function rsvd_core(AC::ITensor, M; svd_kwargs...)
  Q = AC * M
  #@show dims(Q)
  Q = ITensors.qr(Q, commoninds(AC, Q))[1]
  QAC = dag(Q) * AC
  fact = svd(QAC, uniqueind(QAC, AC); svd_kwargs...)
  return fact, Q
end
=#
