from collections import defaultdict
from typing import Any, Dict
from qiskit import QuantumCircuit

def compute_circuit_metrics(circuit: QuantumCircuit) -> Dict[str, Any]:
    """Compute circuit metrics from the provided circuit"""

    # count number of gates by the number of qubits they act on
    counts = defaultdict(int)
    for inst in circuit.data:
        counts[len(inst.qubits)] += 1

    metrics = {
        "circuit_depth": circuit.depth(),
        "circuit_op_counts": circuit.count_ops(),
        "circuit_num_single_qubit_gates": counts[1],
        "circuit_num_gates": circuit.size() - circuit.count_ops().get("measure", 0),
        "circuit_num_qubits": len({q for instr, qargs, _ in circuit.data for q in qargs}),  # using this instead of circuit.num_qubits to count only qubits actually used
    }
    return metrics