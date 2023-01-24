
"""The main object here is `g' a NamedGraph which represents a graphical version of a contraction sequence.
It's vertices describe a partition between the leaves of the sequence (will be labelled with an n = 1 or n = 3 element tuple, where each element of the tuple describes the leaves in one of those partition)
n = 1 implies the vertex is actually a leaf.
Edges connect vertices which are child/ parent and also define a bi-partition"""


"""Function to take a sequence (returned by ITensorNetworks.contraction_sequence) and construct a graph g which represents it (see above)"""
function contraction_sequence_to_graph(contract_sequence)
    
    g = fill_contraction_sequence_graph_vertices(contract_sequence)

    #Now we have the vertices we need to figure out the edges
    for v in vertices(g)
        #Only add edges from a parent (which defines a tripartition and thus has length 3) to its children
        if(length(v) == 3)
        #Work out which vertices it connects to
            concat1, concat2, concat3 =[v[1]..., v[2]...], [v[2]..., v[3]...], [v[1]..., v[3]...]
            for vn in setdiff(vertices(g), [v])
                vn_set = [Set(vni) for vni in vn]
                if(Set(concat1) ∈ vn_set || Set(concat2) ∈ vn_set || Set(concat3) ∈ vn_set)
                    add_edge!(g, v => vn)
                end
            end
        end
    end


    return g
end


function fill_contraction_sequence_graph_vertices(contract_sequence)
    g = NamedGraph()
    leaves = collect(Leaves(contract_sequence))
    fill_contraction_sequence_graph_vertices!(g, contract_sequence[1], leaves)
    fill_contraction_sequence_graph_vertices!(g, contract_sequence[2], leaves)
    return g
end
  
"""Given a contraction sequence which is a subsequence of some larger sequence which is being built on current_g and has leaves `leaves`
Spawn `contract sequence' as a vertex on `current_g' and continue on with its children """
function fill_contraction_sequence_graph_vertices!(g, contract_sequence, leaves)
    if(isa(contract_sequence, Array))
        group1 = collect(Leaves(contract_sequence[1]))
        group2 = collect(Leaves(contract_sequence[2]))
        remaining_verts = setdiff(leaves, vcat(group1, group2))
        add_vertex!(g, (group1, group2, remaining_verts))
        fill_contraction_sequence_graph_vertices!(g, contract_sequence[1], leaves)
        fill_contraction_sequence_graph_vertices!(g, contract_sequence[2], leaves)
    else
        add_vertex!(g, ([contract_sequence], setdiff(leaves, [contract_sequence])))
    end
end

"""Utility functions for the graphical representation of a contraction sequence"""

"""Get the vertex bi-partition that a given edge between non-leaf nodes represents"""
function contraction_tree_leaf_bipartition(g::AbstractGraph, e)

    if(is_leaf_edge(g, e))
        println("ERROR: EITHER THE SOURCE OR THE VERTEX IS A LEAF SO EDGE DOESN'T REALLY REPRESENT A BI-PARTITION")
    end

    vsrc_set, vdst_set = [Set(vni) for vni in src(e)], [Set(vni) for vni in dst(e)]
    c1, c2, c3 = [src(e)[1]..., src(e)[2]...], [src(e)[2]..., src(e)[3]...], [src(e)[1]..., src(e)[3]...]
    left_bipartition = Set(c1) ∈ vdst_set ? c1 : Set(c2) ∈ vdst_set ? c2 : c3

    c1, c2, c3 = [dst(e)[1]..., dst(e)[2]...], [dst(e)[2]..., dst(e)[3]...], [dst(e)[1]..., dst(e)[3]...]
    right_bipartition = Set(c1) ∈ vsrc_set ? c1 : Set(c2) ∈ vsrc_set ? c2 : c3

    return left_bipartition, right_bipartition
end

"""Given a contraction node, get the keys living on all its neighbouring leaves"""
function external_node_keys(g::AbstractGraph, v)
    return [Base.Iterators.flatten(v[findall(==(1), [length(vi) == 1 for vi in v])])...]
end

"""Given a contraction node, get all keys which are not living on a neighbouring leaf"""
function external_contraction_node_ext_keys(g::AbstractGraph, v)
    return [Base.Iterators.flatten(v[findall(==(1), [length(vi) != 1 for vi in v])])...]
end
