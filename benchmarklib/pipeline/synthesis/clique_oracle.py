from tweedledum.bool_function_compiler import QuantumCircuitFunction, circuit_input
from tweedledum import BitVec

@circuit_input(vertices=lambda n: BitVec(n))
def clique_oracle(n: int, k: int, edges) -> BitVec(1):
    """Counts cliques of size 2 in a graph specified by the edge list."""
    s = BitVec(1, 1)  # Start with True (assuming a clique)

    # Check if any non-connected vertices are both selected
    for i in range(n):
        for j in range(i + 1, n):
            # If vertices i and j are both selected (=1) AND there's no edge between them (=0)
            # then it's not a clique
            if edges[i * n + j] == 0:
                s = s & ~(vertices[i] & vertices[j])

    generate_at_least_k_counter(vertices, n, k)

    return s & at_least_k