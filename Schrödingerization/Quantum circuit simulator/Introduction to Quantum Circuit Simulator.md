# Classical Quantum Circuit Simulator

## Introduction
This simulator is designed to efficiently simulate quantum circuits on a classical computer using Torch.tensor operations on GPU or CPU.

Unlike frameworks like Qiskit and Qibo, we do not use a quantum circuit object. Instead, the system size is determined by the input vector, where \( \text{len}(\text{vector}) = 2^n \), with \( n \) being the number of qubits. This approach simplifies the merging of circuits. Furthermore, in actual hardware implementations, there is no quantum circuit object—only instructions for qubits. The gate sequence object serves as one such instruction.

## gate_sequence

`gate_sequence` is a list of dictionary used to store all the operations of a quantum circuit. Each operation (quantum gate) is represented as a dictionary and arranged in sequence.

### Dictionary Structure

Each dictionary contains the following keys:
- **name**: (mandatory) The type of quantum gate and operation
  - "x": Pauli-X gate
  $$X = \begin{bmatrix}   0 & 1 \\ 1 & 0
  \end{bmatrix}$$
  - "y": Pauli-Y gate
  $$Y = \begin{bmatrix}   0 & -i \\ i & 0
  \end{bmatrix}$$
  - "z": Pauli-Z gate
  $$Z = \begin{bmatrix}   1 & 0 \\ 0 & -1
  \end{bmatrix}$$
  - "-z": A Pauli-X gate, followed by a Pauli-Z gate, and then another Pauli-X gate
  $$-Z = \begin{bmatrix}   -1 & 0 \\ 0 & 1
  \end{bmatrix}$$
  - "h": Hadamard gate
  $$H = \frac{1}{\sqrt{2}}\begin{bmatrix}   1 & 1 \\ 1 & -1
  \end{bmatrix}$$
  - "s": S gate
  $$S = \begin{bmatrix}   1 & 0 \\ 0 & i
  \end{bmatrix}$$
  - "s_dagger": The dagger of the S gate
  $$S^\dagger = \begin{bmatrix}   1 & 0 \\ 0 & -i
  \end{bmatrix}$$
  - "rx": Rx gate
  $$R_x(\theta) = \begin{bmatrix}   \cos\frac{\theta}{2} & -i\sin\frac{\theta}{2} \\ -i\sin\frac{\theta}{2} & \cos\frac{\theta}{2}
  \end{bmatrix}$$
  - "ry": Ry gate
  $$R_y(\theta) = \begin{bmatrix}   \cos\frac{\theta}{2} & -\sin\frac{\theta}{2} \\ \sin\frac{\theta}{2} & \cos\frac{\theta}{2}
  \end{bmatrix}$$
  - "rz": Rz gate
  $$R_z(\theta) = \begin{bmatrix}   e^{-i\frac{\theta}{2}} & 0 \\ 0 & e^{i\frac{\theta}{2}}
  \end{bmatrix}$$
  - "phase_gate": Phase gate
  $$P = \begin{bmatrix}   1 & 0 \\ 0 & e^{i\theta}
  \end{bmatrix}$$
  - "phase_gate": Global phase gate
  $$\text{Global phase gate}(\theta) = \begin{bmatrix}   e^{i\theta} & 0 \\ 0 & e^{i\theta}
  \end{bmatrix}$$
  - "u": U gate
  $$U(\theta,\phi,\lambda) = \begin{bmatrix}   \cos\frac{\theta}{2} & -e^{i\lambda}\sin\frac{\theta}{2} \\ e^{i\theta}\sin\frac{\theta}{2} & e^{i(\theta+\lambda)}\cos\frac{\theta}{2}
  \end{bmatrix}$$
  - "swap": Swap two qubits.
  - "create_ancilla": Create a new ancilla qubit with state $\mathinner{|0\rangle}$ and put it at the end of all the qubits.
  - "kill_ancilla": Kill the $\mathinner{|0\rangle}$ state ancilla qubit.
  - "zoom_in": Regard a smaller quantum circuit as a unitary and apply it to some of the qubits in a larger circuit.
- **target**: (optional) A list contains all the indices of the qubits to which the gate is being applied.
- **control**: (optional) A list contains all the indices of the control qubits. If you specified the control, you MUST specify the corresponding control sequence.

- **control_sequence**: (optional) The control state, it should be a list of 0 and 1. This sequence determines when the gate is applied. 
if control is [0,1,2] and control sequence is [0,1,0] it means that the gate will be applied if the qubit 0,1,2 in the state $\mathinner{|010\rangle}$.

- **parameter**: (optional) All the parameters of the gate, it can be the angel of Ry gate, or the name of the ancilla being created.
- **block_gate_sequence**: (optional) The same as the gate_sequence but we can apply a control version of a gate sequence inside the original gate_sequence. See example 4 below. 

### Examples
1. The quantum circuit to create bell state
$$\mathinner{|00\rangle} \rightarrow \frac{\mathinner{|00\rangle}+\mathinner{|11\rangle}}{2}
$$
```python
gate_sequence = []
gate_sequence += [{'name': 'h', 'target': [0]}] #Apply H gate to the 0th qubit
gate_sequence += [{'name': 'x', 'target': [1], 'control': [0], 'control_sequence': [1]}] 
#X gate, qubit 0 is control, qubit 1 is target. Actually, this is exactly a CNOT gate
```
2. The quantum circuit for Quantum Fourier Transform (QFT)
```python
import numpy as np
n = 5 # Number of qubits
gate_sequence = []
for i in range(n):
    gate_sequence += [{'name': 'h', 'target': i}]
    for j in range(i + 1, n):
        gate_sequence += [{'name': 'phase_gate', 'parameter': np.pi/2**(j - i), 'target': i, 'control': [j], 'control_sequence': [1]}]

if n % 2 == 0:
    for i in range(n//2):
        gate_sequence += [{'name': 'swap', 'target': [i, n - 1 - i]}]
else:
    for i in range(int(n/2)):
        gate_sequence += [{'name': 'swap', 'target': [i, n - 1 - i]}]
```
3. Create ancilla and kill ancilla
```python
gate_sequence = []
gate_sequence += [{'name': 'create_ancilla', 'parameter': 'ancilla1'}] #Create an ancilla qubit and name it 'ancilla1'.
"""
Now we created 1 more qubit which mean our vector is enlarged with zeros 
vector = [vector[0],0,vector[1],0,...,vector[2^n-1],0].T

Thus, it is doubled in size, and every other element is zero which correspond to
|vec>|0> state

'ancilla1' is the name of a new qubit. So, now to apply a gate to this qubit we should
put its name as target
targe=['ancilla1']
"""
gate_sequence += [{'name': 'create_ancilla', 'parameter': 'ancilla2'}] #Create an ancilla qubit and name it 'ancilla2'.
gate_sequence += [{'name': 'x', 'target': ['ancilla1']}] #Apply X gate to 'ancilla1'
gate_sequence += [{'name': 'h', 'target': ['ancilla2']}] #Apply H gate to 'ancilla2'
gate_sequence += [{'name': 'h', 'target': ['ancilla2']}]
gate_sequence += [{'name': 'x', 'target': ['ancilla1']}]
'''
x@x=I
H@H=I

Thus, the the ancilla qubits are in zero state at the end of the gate_sequence above

If we do not need those ancilla qubits anymore we can remove them; thus, making our computational easier.
'''

gate_sequence += [{'name': 'kill_ancilla', 'parameter': 'ancilla1'}] #Terminate 'ancilla1', before Terminating, you need to make sure that it is in the |0> state.
gate_sequence += [{'name': 'kill_ancilla', 'parameter': 'ancilla2'}] 
'''
After those two kill_ancilla operations the vector will be decreased in size 4 times

From this moment we cannot address those qubits as target/control qubits, because they do not exist anymore
'''

```
4. Apply a smaller quantum circuit as a controlled unitary operation to a larger one
```python
import numpy as np
gate_sequence_small = [{"name": "ry", "control_sequence": [0], "control": [1], "target": [0], "parameter": np.pi/2}] # Smaller quantum circuit

# Apply H gate to qubit 0
gate_sequence_big = [{"name": "h", "target": [0]}] 
# Apply H gate to qubit 2
gate_sequence_big += [{"name": "h", "target": [2]}] 
# Apply X gate to qubit 3
gate_sequence_big += [{"name": "x", "target": [3]}]  
# Apply Rx gate with angle pi/8 to qubit 4
gate_sequence_big += [{"name": "rx", "target": [4], "parameter": np.pi/8}] 
"""
#Apply the smaller gate sequence to qubits 1 and 0, 
with controlled qubits being 2, 4, and 3, and the controlled state being '011'.

This function recursively use the EXECUTE function but for the small part of the vector
This allows to increase the efficiency of the program.

Inside the zoom in we define a new vector: vector1
len(vector1)=len(vector)/2^3
So, we fix control qubit in the state defined by the control sequence, and for this part of the state
we apply the small_gate_sequence
"""
gate_sequence_big += [{"name":"zoom_in", "block_gate_sequence": gate_sequence_small, "target":[1,0], "control":[2, 4, 3], "control_sequence": [0, 1, 1]}] 
# Apply X gate to qubit 1
gate_sequence_big += [{"name": "x", "target": [1]}] 
"""
#Apply the smaller gate sequence to qubits 2 and 4, 
with controlled qubits being 0 and 1, and the controlled state being 2 ('10').
"""
gate_sequence_big += [{"name":"zoom_in", "block_gate_sequence": gate_sequence_small, "target":[2,4], "control":[0, 1], "control_sequence": 2}] 
#Apply the smaller gate sequence to qubits 3 and 1.
gate_sequence_big += [{"name":"zoom_in", "block_gate_sequence": gate_sequence_small, "target":[3,1]}] 
```

## One_qubit_rotation
Let us consider how to apply a single-qubit unitary to a specific target qubit. We begin with the case where the target qubit is the first qubit (i.e., with index $0$). Suppose the unitary operator is given by
$$
U = \begin{bmatrix}
u_{00} & u_{01} \\
u_{10} & u_{11}
\end{bmatrix},
$$
and is to be applied to an $n$-qubit quantum state represented as
$$
\mathinner{|\psi\rangle} = \begin{bmatrix}
\alpha \\
\beta
\end{bmatrix} \otimes \vec{v} = \begin{bmatrix}
a \vec{v} \\
b \vec{v}
\end{bmatrix} =
\begin{bmatrix}
\vec{w}_0 \\
\vec{w}_1
\end{bmatrix}.
$$
where $\vec{v}$ is a $2^{n-1}$-dimensional vector, and $\vec{w}_0 = \alpha \vec{v}$, $\vec{w}_1 = \beta \vec{v}$ correspond to the components associated with the first qubit being in states $\mathinner{|0\rangle}$ and $\mathinner{|1\rangle}$, respectively.

Applying $U$ to the first qubit transforms the state as follows:
$$
U\mathinner{|\psi\rangle} = \begin{bmatrix}
u_{00}\alpha + u_{01}\beta \\
u_{10}\alpha + u_{11}\beta
\end{bmatrix} \otimes \vec{v}
= \begin{bmatrix}
u_{00}\alpha\vec{v} + u_{01}\beta\vec{v} \\
u_{10}\alpha\vec{v} + u_{11}\beta\vec{v}
\end{bmatrix} = \begin{bmatrix}
u_{00}\vec{w}_0 + u_{01}\vec{w}_1 \\
u_{10}\vec{w}_0 + u_{11}\vec{w}_1
\end{bmatrix}.
$$
Hence, to implement the action of $U$ on the first qubit, it suffices to partition the full state vector into its upper and lower halves, denoted by $\vec{w}_0$ and $\vec{w}_1$, and then update the upper half to $u_{00} \vec{w}_0 + u_{01} \vec{w}_1$ and the lower half to $u_{10} \vec{w}_0 + u_{11} \vec{w}_1$.

To generalise the application of a single-qubit unitary to an arbitrary qubit in an $n$-qubit system, we first reshape the vector $\mathinner{|\psi\rangle}$ to the shape `reshape([2] * n)`. Next, we swap the dimensions corresponding to qubit $0$ and the target qubit using `transpose(0, target)`. The single-qubit operation is then applied to qubit $0$. Afterwards, we reverse the previous transposition by applying `transpose(0, target)` again. Finally, to restore the state to its original vector form, we apply `reshape(-1)`.

```python
import torch
def one_qubit_rotation(n, quantum_state, gate_unitary, target_index):
    """
    :param n: the number of qubits
    :param quantum_state: input quantum state, it should be a 2**n normalized torch vector
    :param gate_unitary: one-qubit gate in unitary matrix form, it should be a 2*2 unitary torch matrix
    :param target_index: the index of the target qubit, it should be an int number
    :return: output quantum state
    """
    if n > 1:
        t1 = quantum_state.reshape([2 for i in range(n)])
        transposed_tensor = t1.transpose(0, target_index)
        target_tensor0 = gate_unitary[0,0] * transposed_tensor[0,:] + gate_unitary[0,1] * transposed_tensor[1,:]
        target_tensor1 = gate_unitary[1,0] * transposed_tensor[0,:] + gate_unitary[1,1] * transposed_tensor[1,:]
        aux_tensor = torch.stack([target_tensor0, target_tensor1], dim=0)
        result_tensor = aux_tensor.transpose(0, target_index)
    else:
        result_tensor = gate_unitary @ quantum_state
    return result_tensor
```

## Control_one_qubit_rotation
In quantum computing, *controlled gates* are operations that are applied to the *target qubits* only when the *control qubits* are in specific states. 
Here, we describe the implementation of a controlled single-qubit gate.

We begin by reshaping the state vector into the form `reshape([2] * n)`. Next, the qubit ordering is rearranged using the `permute` method so that the axes corresponding to the control qubits, 
the target qubit, and all remaining qubits are ordered as: *control_qubits*, *target_qubit*, *other_qubits*.

At this stage, we extract the subvector of the quantum state corresponding to a particular control state using `quantum_substate = quantum_state[control_sequence]`, 
where control_sequence is a tuple of `0`s and `1`s indicating the required control qubit state (e.g., for control state `010`, we have `control_sequence = (0, 1, 0)`). 
The resulting quantum_substate is a vector of size $2^{n - \text{number_of_control_qubit}}$, 
since the dimensions corresponding to the control qubits have been fixed.

Now, since the target qubit has been brought to the 0th position, we can apply the single-qubit gate as a `one_qubit_rotation` operation on `quantum_substate`. 
Finally, the modified substate is written back into the original tensor via `quantum_state[control_sequence]`.

```python
import numpy as np
from fundamental_gates_functions import one_qubit_rotation
def control_one_qubit_rotation(n, quantum_state, gate_unitary, control_indices, control_sequence, target_index):
    """
    :param n: the number of qubits
    :param quantum_state: input quantum state, it should be a 2**n normalized torch vector
    :param gate_unitary: one-qubit gate in unitary matrix form, it should be a 2*2 unitary torch matrix
    :param control_indices: the indices of the control qubits, it should be a list
    :param control_sequence: the control state sequence, it should be a list
    :param target_index: the index of the target qubit, it should be an integer number
    :return:
    """
    n_control = len(control_indices) #The number of the control qubit
    if n_control > 0:
        quantum_state = quantum_state.reshape([2 for i in range(n)])

        #permute the order of the quantum qubits, it will become: control qubits, target qubit, the rest of the qubits
        permutation=tuple(control_indices + [target_index] + [i for i in range(n) if i not in control_indices + [target_index]])
        inverse_permutation=tuple(np.argsort(np.array(permutation)))

        quantum_state = quantum_state.permute(permutation)
        #apply unitary
        vec_0=gate_unitary[0,0] * quantum_state[tuple(control_sequence)][0] + gate_unitary[0,1] * quantum_state[tuple(control_sequence)][1]
        quantum_state[tuple(control_sequence)][1]=(gate_unitary[1,0] * quantum_state[tuple(control_sequence)][0] +
                                                   gate_unitary[1,1] * quantum_state[tuple(control_sequence)][1])
        quantum_state[tuple(control_sequence)][0]=vec_0
        #permute back
        quantum_state = quantum_state.permute(inverse_permutation)
        return quantum_state
    return one_qubit_rotation(n, quantum_state, gate_unitary, target_index)
```

## Authors
Xiajie Huang, Nikita Guseynov