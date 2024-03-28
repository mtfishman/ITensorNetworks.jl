using Test
using Random
using Graphs
using NamedGraphs
using ITensorNetworks
using SplitApplyCombine

using ITensorNetworks: logscalar, scalar, inner, loginner
using ITensors: siteinds, dag

@testset "Inner products, BP vs exact comparison" begin
  Random.seed!(1234)
  L = 4
  χ = 2
  g = NamedGraph(Graphs.SimpleGraph(uniform_tree(L)))
  s = siteinds("S=1/2", g)
  y = randomITensorNetwork(s; link_space=χ)
  x = randomITensorNetwork(s; link_space=χ)

  #First lets do it with the flattened version of the network
  xy = inner_network(x, y)
  xy_scalar = scalar(xy)
  xy_scalar_bp = scalar(xy; alg="bp")
  xy_scalar_logbp = exp(
    logscalar(xy; alg="bp")
  )

  @test xy_scalar ≈ xy_scalar_bp
  @test xy_scalar_bp ≈ xy_scalar_logbp
  @test xy_scalar ≈ xy_scalar_logbp

  #Now lets do it via the inner function
  xy_scalar = inner(x, y; alg="exact")
  xy_scalar_bp = inner(x, y; alg="bp")
  xy_scalar_logbp = exp(loginner(x, y; alg="bp"))

  @test xy_scalar ≈ xy_scalar_bp
  @test xy_scalar_bp ≈ xy_scalar_logbp
  @test xy_scalar ≈ xy_scalar_logbp

#   #test contraction of three layers for expectation values
  A = ITensorNetwork(TTN(ITensorNetworks.heisenberg(s), s))
  xAy_scalar = inner(x', A, y; alg="exact")
  xAy_scalar_bp = inner(x', A, y; alg="bp")
  xAy_scalar_logbp = exp(loginner(x', A, y; alg="bp"))

  @test xAy_scalar ≈ xAy_scalar_bp
  @test xAy_scalar_bp ≈ xAy_scalar_logbp
  @test xAy_scalar ≈ xAy_scalar_logbp
end
nothing
