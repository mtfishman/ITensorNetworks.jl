function ITensors.apply(
  o::ITensor,
  ψ::AbstractITensorNetwork;
  normalize=false,
  ortho=false,
  envs=ITensor[],
  nfullupdatesweeps=10,
  print_fidelity_loss=true,
  envisposdef=false,
  svd_kwargs...
)
  ψ = copy(ψ)
  v⃗ = neighbor_vertices(ψ, o)
  if length(v⃗) == 1
    if ortho
      ψ = orthogonalize(ψ, v⃗[1])
    end
    oψᵥ = apply(o, ψ[v⃗[1]])
    if normalize
      oψᵥ ./= norm(oψᵥ)
    end
    ψ[v⃗[1]] = oψᵥ
  elseif length(v⃗) == 2
    e = v⃗[1] => v⃗[2]
    if !has_edge(ψ, e)
      error("Vertices where the gates are being applied must be neighbors for now.")
    end
    if ortho
      ψ = orthogonalize(ψ, v⃗[1])
    end

    outer_dim_v1, outer_dim_v2 = dim(uniqueinds(ψ[v⃗[1]], o, ψ[v⃗[2]])),
    dim(uniqueinds(ψ[v⃗[2]], o, ψ[v⃗[1]]))
    dim_shared = dim(commoninds(ψ[v⃗[1]], ψ[v⃗[2]]))
    d1, d2 = dim(commoninds(ψ[v⃗[1]], o)), dim(commoninds(ψ[v⃗[2]], o))
    if outer_dim_v1 * outer_dim_v2 <= dim_shared * dim_shared * d1 * d2
      Qᵥ₁, Rᵥ₁ = ITensor(true), copy(ψ[v⃗[1]])
      Qᵥ₂, Rᵥ₂ = ITensor(true), copy(ψ[v⃗[2]])
    else
      Qᵥ₁, Rᵥ₁ = factorize(
        ψ[v⃗[1]], uniqueinds(uniqueinds(ψ[v⃗[1]], ψ[v⃗[2]]), uniqueinds(ψ, v⃗[1]))
      )
      Qᵥ₂, Rᵥ₂ = factorize(
        ψ[v⃗[2]], uniqueinds(uniqueinds(ψ[v⃗[2]], ψ[v⃗[1]]), uniqueinds(ψ, v⃗[2]))
      )
    end

    envs = Vector{ITensor}(envs)
    if !isempty(envs)
      extended_envs = vcat(envs, Qᵥ₁, prime(dag(Qᵥ₁)), Qᵥ₂, prime(dag(Qᵥ₂)))
      Rᵥ₁, Rᵥ₂ = optimise_p_q(
        Rᵥ₁,
        Rᵥ₂,
        extended_envs,
        o;
        nfullupdatesweeps,
        print_fidelity_loss,
        envisposdef,
        svd_kwargs...
      )
    else
      Rᵥ₁, Rᵥ₂ = factorize(
        apply(o, Rᵥ₁ * Rᵥ₂), inds(Rᵥ₁); tags=ITensorNetworks.edge_tag(e), svd_kwargs...
      )
    end

    ψᵥ₁ = Qᵥ₁ * Rᵥ₁
    ψᵥ₂ = Qᵥ₂ * Rᵥ₂

    if normalize
      ψᵥ₁ ./= norm(ψᵥ₁)
      ψᵥ₂ ./= norm(ψᵥ₂)
    end

    ψ[v⃗[1]] = ψᵥ₁
    ψ[v⃗[2]] = ψᵥ₂

  elseif length(v⃗) < 1
    error("Gate being applied does not share indices with tensor network.")
  elseif length(v⃗) > 2
    error("Gates with more than 2 sites is not supported yet.")
  end
  return ψ
end

function ITensors.apply(
  o⃗::Vector{ITensor},
  ψ::AbstractITensorNetwork;
  normalize=false,
  ortho=false,
  svd_kwargs...,
)
  o⃗ψ = ψ
  for oᵢ in o⃗
    o⃗ψ = apply(oᵢ, o⃗ψ; normalize, ortho, svd_kwargs...)
  end
  return o⃗ψ
end

function ITensors.apply(
  o⃗::Scaled,
  ψ::AbstractITensorNetwork;
  normalize=false,
  ortho=false,
  svd_kwargs...,
)
  return maybe_real(Ops.coefficient(o⃗)) *
         apply(Ops.argument(o⃗), ψ; cutoff, maxdim, normalize, ortho, svd_kwargs...)
end

function ITensors.apply(
  o⃗::Prod,
  ψ::AbstractITensorNetwork;
  normalize=false,
  ortho=false,
  svd_kwargs...,
)
  o⃗ψ = ψ
  for oᵢ in o⃗
    o⃗ψ = apply(oᵢ, o⃗ψ; normalize, ortho, svd_kwargs...)
  end
  return o⃗ψ
end

function ITensors.apply(
  o::Op, ψ::AbstractITensorNetwork; normalize=false, ortho=false, svd_kwargs...
)
  return apply(ITensor(o, siteinds(ψ)), ψ; normalize, ortho, svd_kwargs...)
end

### Full Update Routines ###

"""Calculate the overlap of the gate acting on the previous p and q versus the new p and q in the presence of environments. This is the cost function that optimise_p_q will minimise"""
function fidelity(
  envs::Vector{ITensor},
  p_cur::ITensor,
  q_cur::ITensor,
  p_prev::ITensor,
  q_prev::ITensor,
  gate::ITensor,
)
  p_sind, q_sind = commonind(p_cur, gate), commonind(q_cur, gate)
  p_sind_sim, q_sind_sim = sim(p_sind), sim(q_sind)
  gate_sq =
    gate * replaceinds(dag(gate), Index[p_sind, q_sind], Index[p_sind_sim, q_sind_sim])
  term1_tns = vcat(
    [
      p_prev,
      q_prev,
      replaceind(prime(dag(p_prev)), prime(p_sind), p_sind_sim),
      replaceind(prime(dag(q_prev)), prime(q_sind), q_sind_sim),
      gate_sq,
    ],
    envs,
  )
  term1 = ITensors.contract(
    term1_tns; sequence=ITensors.optimal_contraction_sequence(term1_tns)
  )

  term2_tns = vcat(
    [
      p_cur,
      q_cur,
      replaceind(prime(dag(p_cur)), prime(p_sind), p_sind),
      replaceind(prime(dag(q_cur)), prime(q_sind), q_sind),
    ],
    envs,
  )
  term2 = ITensors.contract(
    term2_tns; sequence=ITensors.optimal_contraction_sequence(term2_tns)
  )
  term3_tns = vcat([p_prev, q_prev, prime(dag(p_cur)), prime(dag(q_cur)), gate], envs)
  term3 = ITensors.contract(
    term3_tns; sequence=ITensors.optimal_contraction_sequence(term3_tns)
  )

  f = term3[] / sqrt(term1[] * term2[])
  return f * conj(f)
end

"""Do Full Update Sweeping, Optimising the tensors p and q in the presence of the environments envs,
Specifically this functions find the p_cur and q_cur which optimise envs*gate*p*q*dag(prime(p_cur))*dag(prime(q_cur))"""
function optimise_p_q(
  p::ITensor,
  q::ITensor,
  envs::Vector{ITensor},
  o::ITensor;
  nfullupdatesweeps=10,
  print_fidelity_loss=false,
  envisposdef=true,
  svd_kwargs...
)
  p_cur, q_cur = factorize(apply(o, p * q), inds(p); tags=tags(commonind(p, q)), svd_kwargs...)

  fstart = print_fidelity_loss ? fidelity(envs, p_cur, q_cur, p, q, o) : 0

  qs_ind = setdiff(inds(q_cur), collect(Iterators.flatten(inds.(vcat(envs, p_cur)))))
  ps_ind = setdiff(inds(p_cur), collect(Iterators.flatten(inds.(vcat(envs, q_cur)))))

  opt_b_seq = ITensors.optimal_contraction_sequence(
    vcat(ITensor[p, q, o, dag(prime(q_cur))], envs)
  )
  opt_b_tilde_seq = ITensors.optimal_contraction_sequence(
    vcat(ITensor[p, q, o, dag(prime(p_cur))], envs)
  )
  opt_M_seq = ITensors.optimal_contraction_sequence(
    vcat(ITensor[q_cur, replaceinds(prime(dag(q_cur)), prime(qs_ind), qs_ind), p_cur], envs)
  )
  opt_M_tilde_seq = ITensors.optimal_contraction_sequence(
    vcat(ITensor[p_cur, replaceinds(prime(dag(p_cur)), prime(ps_ind), ps_ind), q_cur], envs)
  )

  function b(
    p::ITensor,
    q::ITensor,
    o::ITensor,
    envs::Vector{ITensor},
    r::ITensor;
    opt_sequence=nothing,
  )
    return noprime(
      ITensors.contract(vcat(ITensor[p, q, o, dag(prime(r))], envs); sequence=opt_sequence)
    )
  end

  function M_p(
    envs::Vector{ITensor},
    p_q_tensor::ITensor,
    s_ind,
    apply_tensor::ITensor;
    opt_sequence=nothing,
  )
    return noprime(
      ITensors.contract(
        vcat(
          ITensor[
            p_q_tensor,
            replaceinds(prime(dag(p_q_tensor)), prime(s_ind), s_ind),
            apply_tensor,
          ],
          envs,
        );
        sequence=opt_sequence,
      ),
    )
  end
  for i in 1:nfullupdatesweeps
    b_vec = b(p, q, o, envs, q_cur; opt_sequence=opt_b_seq)
    M_p_partial = partial(M_p, envs, q_cur, qs_ind; opt_sequence=opt_M_seq)

    p_cur, info = linsolve(
      M_p_partial, b_vec, p_cur; isposdef=envisposdef, ishermitian=false
    )

    b_tilde_vec = b(p, q, o, envs, p_cur; opt_sequence=opt_b_tilde_seq)
    M_p_tilde_partial = partial(M_p, envs, p_cur, ps_ind; opt_sequence=opt_M_tilde_seq)

    q_cur, info = linsolve(
      M_p_tilde_partial, b_tilde_vec, q_cur; isposdef=envisposdef, ishermitian=false
    )
  end

  fend = print_fidelity_loss ? fidelity(envs, p_cur, q_cur, p, q, o) : 0

  diff = real(fend - fstart)
  if print_fidelity_loss && diff < -eps(diff) && nfullupdatesweeps >= 1
    println(
      "Warning: Krylov Solver Didn't Find a Better Solution by Sweeping. Something might be amiss.",
    )
  end

  return p_cur, q_cur
end

partial = (f, a...; c...) -> (b...) -> f(a..., b...; c...)

"""Vidal Gauge Apply()"""
function apply_vidal_itn(
  ψ::AbstractITensorNetwork,
  bond_tensors::DataGraph,
  o::Union{ITensor, Nothing};
  regularization=10 * eps(real(scalartype(ψ))),
  normalize = false,
  edge_to_act_on = first(edges(ψ)),
  svd_kwargs...
)
  ψ = copy(ψ)
  bond_tensors = copy(bond_tensors)
  v⃗ = o == nothing ? [src(edge_to_act_on), dst(edge_to_act_on)] : neighbor_vertices(ψ, o)
  if length(v⃗) == 2
    e = NamedEdge(v⃗[1] => v⃗[2])
    ψv1, ψv2 = copy(ψ[src(e)]), copy(ψ[dst(e)])
    e_ind = commonind(ψv1, ψv2)

    for vn in neighbors(ψ, src(e))
      if (vn != dst(e))
        ψv1 = noprime(ψv1 * bond_tensors[vn => src(e)])
      end
    end

    for vn in neighbors(ψ, dst(e))
      if (vn != src(e))
        ψv2 = noprime(ψv2 * bond_tensors[vn => dst(e)])
      end
    end

    ψv2 = noprime(ψv2 * bond_tensors[e])

    if o != nothing
      G1, G2 = factorize(o, Index[commonind(ψv1, o), commonind(ψv1, o)']; cutoff = 1e-16)
      ψv1 = noprime(ψv1 * G1)
      ψv2 = noprime(ψv2 * G2)
    end

    Qᵥ₁, Rᵥ₁ = factorize(ψv1, uniqueinds(ψv1, ψv2); cutoff = 1e-16)
    Qᵥ₂, Rᵥ₂ = factorize(ψv2, uniqueinds(ψv2, ψv1); cutoff = 1e-16)

    theta = Rᵥ₁ * Rᵥ₂

    U, S, V = ITensors.svd(theta, uniqueinds(Rᵥ₁, Rᵥ₂); lefttags = ITensorNetworks.edge_tag(e), righttags = ITensorNetworks.edge_tag(e), svd_kwargs...)

    ind_to_replace = commonind(V, S)
    ind_to_replace_with = commonind(U, S)
    replaceind!(S, ind_to_replace, ind_to_replace_with')
    replaceind!(V, ind_to_replace, ind_to_replace_with)


    ψv1, bond_tensors[e], ψv2 = U * Qᵥ₁, S, V * Qᵥ₂


    for vn in neighbors(ψ, src(e))
      if (vn != dst(e))
        ψv1 = noprime(ψv1 * inv_diag(bond_tensors[vn => src(e)]))
      end
    end

    for vn in neighbors(ψ, dst(e))
      if (vn != src(e))
        ψv2 = noprime(ψv2 * inv_diag(bond_tensors[vn => dst(e)]))
      end
    end

    if normalize
      normalize!(ψv1)
      normalize!(ψv2)
      normalize!(bond_tensors[e])
    end

    ψ[src(e)], ψ[dst(e)] = ψv1, ψv2

    return ψ, bond_tensors

  else
    ψ = apply(o, ψ)
    return ψ, bond_tensors
  end

end