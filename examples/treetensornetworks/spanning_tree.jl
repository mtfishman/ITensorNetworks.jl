using NamedGraphs
using ITensors
using ITensorNetworks
using ITensorUnicodePlots

s = siteinds("S=1/2", named_grid((4, 4)))
ψ = ITensorNetwork(s; link_space=3)

@visualize ψ

# Gives a snake pattern
t_dfs = dfs_tree(ψ, (1, 1))

@visualize t_dfs

# Gives a comb pattern
t_bfs = bfs_tree(ψ, (1, 1))

@visualize t_bfs

nothing
