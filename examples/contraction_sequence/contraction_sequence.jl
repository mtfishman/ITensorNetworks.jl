using NamedGraphs
using ITensors
using ITensorNetworks
using Random

Random.seed!(1234)

ITensors.disable_warn_order()

system_dims = (2, 3)
g = named_grid(system_dims)
s = siteinds("S=1/2", g)

χ = 10
ψ = randomITensorNetwork(s; link_space=χ)

ψψ = norm_sqr_network(ψ)

# Contraction sequence for exactly computing expectation values
# contract_edges = map(t -> (1, t...), collect(keys(cartesian_to_linear(system_dims))))
# inner_sequence = reduce((x, y) -> [x, y], contract_edges)

println("optimal")
seq_optimal = @time contraction_sequence(ψψ; alg="optimal")

using OMEinsumContractionOrders

println("greedy")
seq_greedy = @time contraction_sequence(ψψ; alg="greedy")
res_greedy = @time contract(ψψ; sequence=seq_greedy)

println("tree_sa")
seq_tree_sa = @time contraction_sequence(ψψ; alg="tree_sa")

println("sa_bipartite")
seq_sa_bipartite = @time contraction_sequence(ψψ; alg="sa_bipartite")

using KaHyPar

println("kahypar_bipartite")
seq_kahypar_bipartite = @time contraction_sequence(
  ψψ; alg="kahypar_bipartite", sc_target=200
)
