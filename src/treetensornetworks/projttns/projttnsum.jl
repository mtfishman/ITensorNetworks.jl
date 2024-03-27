using ITensors: ITensors, contract
using ITensors.LazyApply: LazyApply, terms
using NamedGraphs: NamedGraphs, incident_edges

"""
ProjTTNSum
"""
struct ProjTTNSum{V,T<:AbstractProjTTN{V},Z<:Number} <: AbstractProjTTN{V}
  terms::Vector{T}
  factors::Vector{Z}
  function ProjTTNSum(terms::Vector{<:AbstractProjTTN}, factors::Vector{<:Number})
    return new{vertextype(eltype(terms)),eltype(terms),eltype(factors)}(terms, factors)
  end
end

LazyApply.terms(P::ProjTTNSum) = P.terms
factors(P::ProjTTNSum) = P.factors

Base.copy(P::ProjTTNSum) = ProjTTNSum(copy.(terms(P)), copy(factors(P)))

function ProjTTNSum(operators::Vector{<:AbstractProjTTN})
  return ProjTTNSum(operators, fill(one(Bool), length(operators)))
end
function ProjTTNSum(operators::Vector{<:AbstractTTN})
  return ProjTTNSum(ProjTTN.(operators))
end

on_edge(P::ProjTTNSum) = on_edge(terms(P)[1])

ITensorMPS.nsite(P::ProjTTNSum) = nsite(terms(P)[1])

function set_nsite(Ps::ProjTTNSum, nsite)
  return ProjTTNSum(map(p -> set_nsite(p, nsite), terms(Ps)), factors(Ps))
end

DataGraphs.underlying_graph(P::ProjTTNSum) = underlying_graph(terms(P)[1])

Base.length(P::ProjTTNSum) = length(terms(P)[1])

sites(P::ProjTTNSum) = sites(terms(P)[1])

NamedGraphs.incident_edges(P::ProjTTNSum) = incident_edges(terms(P)[1])

internal_edges(P::ProjTTNSum) = internal_edges(terms(P)[1])

ITensors.product(P::ProjTTNSum, v::ITensor) = noprime(contract(P, v))

function ITensors.contract(P::ProjTTNSum, v::ITensor)
  res = mapreduce(+, zip(factors(P), terms(P))) do (f, p)
    f * contract(p, v)
  end
  return res
end

function contract_ket(P::ProjTTNSum, v::ITensor)
  res = mapreduce(+, zip(factors(P), terms(P))) do (f, p)
    f * contract_ket(p, v)
  end
  return res
end

function Base.eltype(P::ProjTTNSum)
  return mapreduce(eltype, promote_type, terms(P))
end

(P::ProjTTNSum)(v::ITensor) = product(P, v)

Base.size(P::ProjTTNSum) = size(terms(P)[1])

function position(P::ProjTTNSum, psi::AbstractTTN, pos)
  theterms = map(M -> position(M, psi, pos), terms(P))
  #@show typeof(theterms)
  #@show factors(P)
  return ProjTTNSum(theterms, factors(P))
end
