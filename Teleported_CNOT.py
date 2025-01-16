"""Code for Teleported CNOT gate.

This module defines code for Teleported(non-local) CNOT gate protocol for bell state entanglement.
"""

from sequence.components.circuit import Circuit


def teleported_cnot_circuit():
    """Method to apply single-qubit Hadamard gate on a qubit.

    Teleported CNOT circuit representation:

    s1  |a>-------------|--------|Z|---|a>             
                        |         |            
    c1  |0>---|H|---|---o---<Z|   |
                    |        |    |              
    c2  |0>---------o---|----|---<X|
                        |    |               
    s2  |b>-------------o---|X|-------|b⊕a>

        """    
    qc = Circuit(size=4)

    # Assign labels for clarity: c1, c2 (communication qubits), s1, s2 (storage qubits)
    c1, c2, s1, s2 = 0, 1, 2, 3

    # Step 1: Prepare a Bell state on c1 and c2
    qc.h(c1) 
    qc.cx(c1, c2) 

    # Step 2: Perform CNOT between s1 and c1 (local gate)
    qc.cx(s1, c1)
    qc.cx(c2, s2)

    # Step 4: Measure c1 in Z basis and c2 in X basis
    qc.measure(c1)  
    qc.measure(c2)  

    qc.x(s2)
    qc.z(s1)

    return qc

# Create the circuits
teleported_cnot = teleported_cnot_circuit()

print("Teleported CNOT Circuit:")
print(teleported_cnot.serialize())