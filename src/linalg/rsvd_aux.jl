function get_column_space(A::Vector{<:ITensor}, lc::ITensors.QNIndex, rc::ITensors.QNIndex)
  #gets non-zero blocks in rc by sticking in lc and contracting through
  viable_sectors = Vector{Pair{QN,Int64}}()
  for s in space(lc)
    qn = first(s)
    trial = dag(randomITensor(qn, lc))
    adtrial = foldl(*, A; init=trial)
    nnzblocks(adtrial) == 0 && continue
    @assert nnzblocks(adtrial) == 1
    i = Int(nzblocks(adtrial)[1])
    thesector = space(only(inds(adtrial)))[i]
    push!(viable_sectors, thesector)
  end
  return viable_sectors
end

function get_column_space(A::Vector{<:ITensor}, lc::Index{Int64}, rc::Index{Int64})
  #gets non-zero blocks in rc by sticking in lc and contracting through
  viable_sectors = Vector{Pair{Int64,Int64}}()
  !any(isempty.(A)) && (viable_sectors = [dim(rc) => dim(rc)])
  return viable_sectors
end

# remove since code passes through vector of ITensors
#=
function get_column_space(A::ITensor, lc::Index,rc::Index)
  #gets non-zero blocks in rc by sticking in lc and contracting through
  viable_sectors=Vector{Pair{QN,Int64}}()
  ind_loc=only(findall(isequal(rc),inds(A)))
  unique_qns=unique(qns(A,ind_loc))
  for s in space(rc)
    qn = first(s)
    qn in unique_qns || continue 
    push!(viable_sectors, s)
  end
  return viable_sectors
end
=#

qns(t::ITensor) = qns(collect(eachnzblock(t)), t)
qns(t::ITensor, i::Int) = qns(collect(eachnzblock(t)), t, i)

function qns(bs::Vector{Block{n}}, t::ITensor) where {n}
  return [qns(b, t) for b in bs]
end

function qns(bs::Vector{Block{n}}, t::ITensor, i::Int) where {n}
  return [qns(b, t)[i] for b in bs]
end

function qns(b::Block{n}, t::ITensor) where {n}
  theqns = QN[]
  for i in 1:order(t)
    theb = getindex(b, i)
    theind = inds(t)[i]
    push!(theqns, ITensors.qn(space(theind), Int(theb)))
  end
  return theqns
end

#QN version
function build_guess_matrix(
  eltype::Type{<:Number}, ind, sectors::Vector{Pair{QN,Int64}}, n::Int, p::Int
)
  aux_spaces = Pair{QN,Int64}[]
  for s in sectors
    thedim = last(s)
    qn = first(s)
    en = min(n + p, thedim)
    push!(aux_spaces, Pair(qn, en))
  end
  aux_ind = Index(aux_spaces; dir=dir(ind))
  M = randomITensor(eltype, dag(ind), aux_ind)    #defaults to zero flux
  @assert nnzblocks(M) != 0
  return M
end

#non-QN version
function build_guess_matrix(
  eltype::Type{<:Number}, ind, sectors::Vector{Pair{Int,Int64}}, n::Int, p::Int
)
  thedim = dim(ind)
  en = min(n + p, thedim)
  aux_ind = Index(en)
  M = randomITensor(eltype, ind, aux_ind)
  return M
end

#non-QN version
function build_guess_matrix(
  eltype::Type{<:Number}, ind, sectors::Vector{Pair{Int,Int}}, ndict::Dict;
) ##given ndict, sectors is not necessary here
  thedim = dim(ind)
  en = min(ndict[space(ind)], thedim)
  aux_ind = Index(en)
  M = randomITensor(eltype, ind, aux_ind)
  return M
end

#QN version
function build_guess_matrix(
  eltype::Type{<:Number}, ind, sectors::Vector{Pair{QN,Int64}}, ndict::Dict;
) ##given ndict, sectors is not necessary here
  aux_spaces = Pair{QN,Int64}[]
  for s in sectors
    thedim = last(s)
    qn = first(s)
    en = min(ndict[qn], thedim)
    push!(aux_spaces, Pair(qn, en))
  end
  aux_ind = Index(aux_spaces; dir=dir(ind))
  M = randomITensor(eltype, dag(ind), aux_ind)
  @assert nnzblocks(M) != 0
  return M
end

# non-QN version
function init_guess_sizes(cind, sectors::Nothing, n::Int, rule)
  thedim = dim(cind)
  ndict = Dict{Int64,Int64}()
  ndict[thedim] = min(thedim, rule(n))
  return ndict
end

# QN version
function init_guess_sizes(cind, sectors::Vector{Pair{QN,Int64}}, n::Int, rule)
  @assert hasqns(cind)
  ndict = Dict{QN,Int64}()
  for s in sectors
    thedim = last(s)
    qn = first(s)
    ndict[qn] = min(rule(n), thedim)
  end
  return ndict
end

function increment_guess_sizes(ndict::Dict{QN,Int64}, n_inc::Int, rule)
  for key in keys(ndict)
    ndict[key] = ndict[key] + rule(n_inc)
  end
  return ndict
end

function increment_guess_sizes(ndict::Dict{Int64,Int64}, n_inc::Int, rule)
  for key in keys(ndict)
    ndict[key] = ndict[key] + rule(n_inc)
  end
  return ndict
end

function approx_the_same(o, n; abs_eps=1e-12, rel_eps=1e-6)
  absdev = abs.(o .- n)
  reldev = abs.((o .- n) ./ n)
  abs_conv = absdev .< abs_eps
  rel_conv = reldev .< rel_eps
  return all(abs_conv .|| rel_conv)
end

function is_converged_block(o, n; svd_kwargs...)
  maxdim = get(svd_kwargs, :maxdim, Inf)
  if length(o) != length(n)
    return false
  else
    r = min(maxdim, length(o))
    #ToDo: Pass kwargs?
    return approx_the_same(o[1:r], n[1:r])
  end
end

function is_converged!(ndict, old_fact, new_fact; n_inc=1, has_qns=true, svd_kwargs...)
  oU, oS, oV = old_fact.U, old_fact.S, old_fact.V
  nU, nS, nV = new_fact.U, new_fact.S, new_fact.V

  # check for non-existing tensor
  (isempty(nS) && isempty(oS)) && return true
  (isempty(nS) && !(isempty(oS))) && warning("New randomized SVD is empty
  while prior iteration was not. It is very unlikely that this is correct.
  Exiting.")

  theflux = flux(nS)
  oldflux = flux(oS)

  if !has_qns
    if norm(oS) == 0.0
      if norm(nS) == 0.0
        return true
      else
        ndict[first(keys(ndict))] *= 2
        return false
      end
    end
  end

  maxdim = get(svd_kwargs, :maxdim, Inf)
  # deal with non QN tensors first and a simple case for blocksparse
  os = sort(storage(oS); rev=true)
  ns = sort(storage(nS); rev=true)
  if length(os) >= maxdim && length(ns) >= maxdim
    conv_global = approx_the_same(os[1:maxdim], ns[1:maxdim])
  elseif length(os) != length(ns)
    conv_global = false
  else
    r = length(ns)
    conv_global = approx_the_same(os[1:r], ns[1:r]) #shouldn't this error for os shorter than ns?
  end
  if !hasqns(oS)
    if conv_global == false
      #ndict has only one key 
      ndict[first(keys(ndict))] *= 2
    end
    return conv_global
  end
  conv_bool_total = true

  # deal with QN tensors now
  ncrind = commonind(nS, nV)
  ocrind = commonind(oS, oV)
  n_ind_loc = only(findall(isequal(ncrind), inds(nS)))
  nqns = first.(space(ncrind))
  oqns = first.(space(ocrind))

  for qn in keys(ndict)
    if !(qn in nqns) && !(qn in oqns)
      conv_bool = true
      continue
      #qn
    elseif !(qn in nqns)
      ndict[qn] *= 2
      conv_bool_total = false
      warn("QN was in old factorization but not in new one, this shouldn't happen often!")
      continue
    elseif !(qn in oqns)
      ndict[qn] *= 2
      conv_bool_total = false
      continue
    end
    #qn present in both old and new factorization, grab singular values to compare
    ovals = get_qnblock_vals(qn, ocrind, oS)
    nvals = get_qnblock_vals(qn, ncrind, nS)
    conv_bool = is_converged_block(collect(ovals), collect(nvals); svd_kwargs...)
    if conv_bool == false
      ndict[qn] *= 2
    end
    conv_bool_total *= conv_bool
  end
  if conv_bool_total == true
    @assert conv_global == true
  else
    if conv_global == true
      println(
        "Subspace expansion, rand. svd: singular vals converged globally, but may not be optimal, doing another iteration",
      )
    end
  end
  return conv_bool_total::Bool
end

"""Extracts (singular) values in block associated with `qn` in `ind`` from diagonal ITensor `svalt``."""
function get_qnblock_vals(qn, ind, svalt)
  s = space(ind)
  ind_loc = only(findall(isequal(ind), inds(svalt)))
  theblocks = eachnzblock(svalt)
  blockdict = Int.(getindex.(theblocks, ind_loc))
  qnindtoblock = Dict(collect(values(blockdict)) .=> collect(keys(blockdict))) #refactor

  qnind = findfirst((first.(s)) .== [qn]) # refactor
  theblock = qnindtoblock[qnind]
  vals = diag(svalt[theblock])

  return vals
end

"""Check that the factorization isn't empty. / both empty"""
function _sanity_checks()
  return nothing
end
