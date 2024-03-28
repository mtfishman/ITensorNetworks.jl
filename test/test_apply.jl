using ITensorNetworks
using ITensorNetworks:
  environment, update, inner, norm_sqr_network, BeliefPropagationCache, VidalITensorNetwork
using Test
using Compat
using ITensors
using Metis
using NamedGraphs
using Random
using LinearAlgebra
using SplitApplyCombine

@testset "apply" begin
  Random.seed!(5623)
  g_dims = (2, 3)
  n = prod(g_dims)
  g = named_grid(g_dims)
  s = siteinds("S=1/2", g)
  χ = 2
  ψ = randomITensorNetwork(s; link_space=χ)
  v1, v2 = (2, 2), (1, 2)
  ψψ = norm_sqr_network(ψ)

  #Simple Belief Propagation Grouping
  bp_cache = BeliefPropagationCache(ψψ, group(v -> v[1], vertices(ψψ)))
  bp_cache = update(bp_cache; maxiter=20)
  envsSBP = environment(bp_cache, PartitionVertex.([v1, v2]))

  ψv = VidalITensorNetwork(ψ)

  #This grouping will correspond to calculating the environments exactly (each column of the grid is a partition)
  bp_cache = BeliefPropagationCache(ψψ, group(v -> v[1][1], vertices(ψψ)))
  bp_cache = update(bp_cache; maxiter=20)
  envsGBP = environment(bp_cache, [(v1, 1), (v1, 2), (v2, 1), (v2, 2)])

  ngates = 5

  for i in 1:ngates
    o = ITensors.op("RandomUnitary", s[v1]..., s[v2]...)

    ψOexact = apply(o, ψ; cutoff=1e-16)
    ψOSBP = apply(
      o,
      ψ;
      envs=envsSBP,
      maxdim=χ,
      normalize=true,
      print_fidelity_loss=true,
      envisposdef=true,
    )
    ψOv = apply(o, ψv; maxdim=χ, normalize=true)
    ψOVidal_symm = ITensorNetwork(ψOv)
    ψOGBP = apply(
      o,
      ψ;
      envs=envsGBP,
      maxdim=χ,
      normalize=true,
      print_fidelity_loss=true,
      envisposdef=true,
    )
    fSBP = inner(ψOSBP, ψOexact) / sqrt(inner(ψOexact, ψOexact) * inner(ψOSBP, ψOSBP))
    fVidal =
      inner(ψOVidal_symm, ψOexact) /
      sqrt(inner(ψOexact, ψOexact) * inner(ψOVidal_symm, ψOVidal_symm))
    fGBP = inner(ψOGBP, ψOexact) / sqrt(inner(ψOexact, ψOexact) * inner(ψOGBP, ψOGBP))

    @test real(fGBP * conj(fGBP)) >= real(fSBP * conj(fSBP))

    @test isapprox(real(fSBP * conj(fSBP)), real(fVidal * conj(fVidal)); atol=1e-3)
  end
end
