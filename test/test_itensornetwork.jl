using Dictionaries
using Distributions
using GraphsFlows
using ITensors
using ITensorNetworks
using NamedGraphs
using Random
using Test

using ITensorNetworks: norm_sqr, neighbor_itensors

@testset "ITensorNetwork tests" begin
  @testset "ITensorNetwork Basics" begin
    Random.seed!(1234)
    g = named_grid((4,))
    s = siteinds("S=1/2", g)

    @test s isa IndsNetwork
    @test nv(s) == 4
    @test ne(s) == 3
    @test neighbors(s, (2,)) == [(1,), (3,)]

    tn = ITensorNetwork(s; link_space=2)

    @test nv(tn) == 4
    @test ne(tn) == 3
    @test tn isa ITensorNetwork
    @test neighbors(tn, (2,)) == [(1,), (3,)]

    # TODO: How to support this syntax?
    @test_broken tn[1] isa ITensor
    @test tn[(1,)] isa ITensor

    # TODO: How to support this syntax?
    @test_broken order(tn[1]) == 2
    @test order(tn[(1,)]) == 2

    # TODO: How to support this syntax?
    @test_broken tn[2] isa ITensor
    @test tn[(2,)] isa ITensor

    # TODO: How to support this syntax?
    @test_broken order(tn[2]) == 3
    @test order(tn[(2,)]) == 3

    # XXX: Slicing syntax is no long supported, use `induced_subgraph`.
    @test_broken tn[1:2] isa ITensorNetwork
    # TODO: Support this syntax, maybe rename `subgraph`.
    @test_broken induced_subgraph(tn, [(1,), (2,)]) isa ITensorNetwork

    randn!.(vertex_data(tn))
    tn′ = sim(dag(tn); sites=[])

    @test tn′ isa ITensorNetwork
    inner_tn = tn ⊗ tn′
    @test inner_tn isa ITensorNetwork
    sequence = contraction_sequence(inner_tn)
    @test sequence isa Vector
    inner_res = contract(inner_tn; sequence)[]
    @test inner_res isa Float64

    # test that by default vertices are linked by bond-dimension 1 index
    tn = ITensorNetwork(s)
    @test isone(ITensors.dim(commonind(tn[(1,)], tn[(2,)])))
  end

  @testset "Constructors from ITensors" begin
    i, j, k, l = Index.(fill(2, 4))
    A = ITensor(i, j)
    B = ITensor(j, k)
    C = ITensor(k, l)

    tn = ITensorNetwork([A, B, C])
    @test issetequal(vertices(tn), [1, 2, 3])
    @test issetequal(edges(tn), NamedEdge.([1 => 2, 2 => 3]))

    tn = ITensorNetwork(["A", "B", "C"], [A, B, C])
    @test issetequal(vertices(tn), ["A", "B", "C"])
    @test issetequal(edges(tn), NamedEdge.(["A" => "B", "B" => "C"]))

    tn = ITensorNetwork(["A" => A, "B" => B, "C" => C])
    @test issetequal(vertices(tn), ["A", "B", "C"])
    @test issetequal(edges(tn), NamedEdge.(["A" => "B", "B" => "C"]))
  end

  @testset "Contract edge (regression test for issue #5)" begin
    dims = (2, 2)
    g = named_grid(dims)
    s = siteinds("S=1/2", g)
    ψ = ITensorNetwork(s, v -> "↑")
    tn = inner_network(ψ, ψ)
    tn_2 = contract(tn, ((1, 2), 2) => ((1, 2), 1))
    @test !has_vertex(tn_2, ((1, 2), 2))
    @test tn_2[((1, 2), 1)] ≈ tn[((1, 2), 2)] * tn[((1, 2), 1)]
  end

  @testset "Remove edge (regression test for issue #5)" begin
    dims = (2, 2)
    g = named_grid(dims)
    s = siteinds("S=1/2", g)
    ψ = ITensorNetwork(s, v -> "↑")
    rem_vertex!(ψ, (1, 2))
    tn = inner_network(ψ, ψ)
    @test !has_vertex(tn, ((1, 2), 1))
    @test !has_vertex(tn, ((1, 2), 2))
    @test has_vertex(tn, ((1, 1), 1))
    @test has_vertex(tn, ((1, 1), 2))
    @test has_vertex(tn, ((2, 1), 1))
    @test has_vertex(tn, ((2, 1), 2))
    @test has_vertex(tn, ((2, 2), 1))
    @test has_vertex(tn, ((2, 2), 2))
  end

  @testset "Custom element type" for eltype in (Float32, Float64, ComplexF32, ComplexF64),
    link_space in (nothing, 3),
    g in (
      grid((4,)),
      named_grid((3, 3)),
      siteinds("S=1/2", grid((4,))),
      siteinds("S=1/2", named_grid((3, 3))),
    )

    ψ = ITensorNetwork(g; link_space) do v, inds...
      return itensor(randn(eltype, dims(inds)...), inds...)
    end
    @test Base.eltype(ψ[first(vertices(ψ))]) == eltype
    ψ = ITensorNetwork(g; link_space) do v, inds...
      return itensor(randn(dims(inds)...), inds...)
    end
    @test Base.eltype(ψ[first(vertices(ψ))]) == Float64
    ψ = randomITensorNetwork(eltype, g; link_space)
    @test Base.eltype(ψ[first(vertices(ψ))]) == eltype
    ψ = randomITensorNetwork(g; link_space)
    @test Base.eltype(ψ[first(vertices(ψ))]) == Float64
    ψ = ITensorNetwork(eltype, undef, g; link_space)
    @test Base.eltype(ψ[first(vertices(ψ))]) == eltype
    ψ = ITensorNetwork(undef, g)
    @test Base.eltype(ψ[first(vertices(ψ))]) == Float64
  end

  @testset "randomITensorNetwork with custom distributions" begin
    distribution = Uniform(-1.0, 1.0)
    tn = randomITensorNetwork(distribution, named_grid(4); link_space=2)
    # Note: distributions in package `Distributions` currently doesn't support customized
    # eltype, and all elements have type `Float64`
    @test Base.eltype(tn[first(vertices(tn))]) == Float64
  end

  @testset "orthogonalize" begin
    tn = randomITensorNetwork(named_grid(4); link_space=2)
    Z = norm_sqr(tn)

    tn_ortho = factorize(tn, 4 => 3)

    # TODO: Error here in arranging the edges. Arrange by hash?
    Z̃ = norm_sqr(tn_ortho)
    @test nv(tn_ortho) == 5
    @test nv(tn) == 4
    @test Z ≈ Z̃

    tn_ortho = orthogonalize(tn, 4 => 3)
    Z̃ = norm_sqr(tn_ortho)
    @test nv(tn_ortho) == 4
    @test nv(tn) == 4
    @test Z ≈ Z̃

    tn_ortho = orthogonalize(tn, 1)
    Z̃ = norm_sqr(tn_ortho)
    @test Z ≈ Z̃
    Z̃ = inner(tn_ortho, tn)
    @test Z ≈ Z̃
  end

  @testset "dijkstra_shortest_paths" begin
    tn = ITensorNetwork(named_grid(4); link_space=2)
    paths = dijkstra_shortest_paths(tn, [1])
    @test paths.dists == Dictionary([0, 1, 2, 3])
    @test paths.parents == Dictionary([1, 1, 2, 3])
    @test paths.pathcounts == Dictionary([1.0, 1.0, 1.0, 1.0])
  end

  @testset "mincut" begin
    tn = ITensorNetwork(named_grid(4); link_space=3)
    w = weights(tn)
    @test w isa Dictionary{Tuple{Int,Int},Float64}
    @test length(w) ≈ ne(tn)
    @test w[(1, 2)] ≈ log2(3)
    @test w[(2, 3)] ≈ log2(3)
    @test w[(3, 4)] ≈ log2(3)
    p1, p2, wc = GraphsFlows.mincut(tn, 2, 3)
    @test issetequal(p1, [1, 2])
    @test issetequal(p2, [3, 4])
    @test isone(wc)

    p1, p2, wc = GraphsFlows.mincut(tn, 2, 3, w)
    @test issetequal(p1, [1, 2])
    @test issetequal(p2, [3, 4])
    @test wc ≈ log2(3)
  end

  @testset "Index access" begin
    dims = (2, 2)
    g = named_grid(dims)
    s = siteinds("S=1/2", g)
    ψ = ITensorNetwork(s; link_space=2)

    nt = neighbor_itensors(ψ, (1, 1))
    @test length(nt) == 2
    @test all(map(hascommoninds(ψ[1, 1]), nt))

    @test all(map(t -> isempty(commoninds(inds(t), uniqueinds(ψ, (1, 1)))), nt))

    e = (1, 1) => (2, 1)
    uie = uniqueinds(ψ, e)
    @test isempty(commoninds(uie, inds(ψ[2, 1])))
    @test issetequal(uie, union(commoninds(ψ[1, 1], ψ[1, 2]), uniqueinds(ψ, (1, 1))))

    @test siteinds(ψ, (1, 1)) == s[1, 1]

    cie = commoninds(ψ, e)
    @test hasinds(ψ[1, 1], cie) && hasinds(ψ[2, 1], cie)
    @test isempty(commoninds(uie, cie))

    @test linkinds(ψ, e) == commoninds(ψ[1, 1], ψ[2, 1])

    @test length(externalinds(ψ)) == length(vertices(g))
    @test length(internalinds(ψ)) == length(edges(g))
  end

  @testset "ElType conversion, $new_eltype" for new_eltype in (Float32, ComplexF64)
    dims = (2, 2)
    g = named_grid(dims)
    s = siteinds("S=1/2", g)
    ψ = randomITensorNetwork(s; link_space=2)
    @test ITensors.scalartype(ψ) == Float64

    ϕ = ITensors.convert_leaf_eltype(new_eltype, ψ)
    @test ITensors.scalartype(ϕ) == new_eltype
  end

  @testset "Construction from state map" for ElT in (Float32, ComplexF64)
    dims = (2, 2)
    g = named_grid(dims)
    s = siteinds("S=1/2", g)
    state_map(v::Tuple) = iseven(sum(isodd.(v))) ? "↑" : "↓"

    ψ = ITensorNetwork(s, state_map)
    t = ψ[2, 2]
    si = only(siteinds(ψ, (2, 2)))
    bi = map(e -> only(linkinds(ψ, e)), incident_edges(ψ, (2, 2)))
    @test eltype(t) == Float64
    @test abs(t[si => "↑", [b => end for b in bi]...]) == 1.0 # insert_links introduces extra signs through factorization...
    @test t[si => "↓", [b => end for b in bi]...] == 0.0

    ϕ = ITensorNetwork(ElT, s, state_map)
    t = ϕ[2, 2]
    si = only(siteinds(ϕ, (2, 2)))
    bi = map(e -> only(linkinds(ϕ, e)), incident_edges(ϕ, (2, 2)))
    @test eltype(t) == ElT
    @test abs(t[si => "↑", [b => end for b in bi]...]) == convert(ElT, 1.0) # insert_links introduces extra signs through factorization...
    @test t[si => "↓", [b => end for b in bi]...] == convert(ElT, 0.0)
  end

  @testset "Priming and tagging" begin
    # TODO: add actual tests

    tooth_lengths = fill(2, 3)
    c = named_comb_tree(tooth_lengths)
    is = siteinds("S=1/2", c)
    tn = randomITensorNetwork(is; link_space=3)
    @test_broken swapprime(tn, 0, 2)
  end
end
