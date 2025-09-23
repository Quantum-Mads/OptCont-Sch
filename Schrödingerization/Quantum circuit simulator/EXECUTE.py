from fundamental_gates_functions import control_one_qubit_rotation, create_qubit, kill_qubit, zoom_in_permutation, zoom_out_permutation, kill_nonzero_qubit, swap
from tqdm import tqdm
import torch
import numpy as np
from fundamental_gates_unitary import gate_set
from fundamental_digital_functions import binary_form_of_integer

def EXECUTE(gate_sequence, quantum_state):
    """
    This function executes the gate sequence and returns the resulting quantum state.
    :param gate_sequence: The gate circuit information
    :param quantum_state: The initial quantum state (torch.Tensor on CUDA)
    :return: The resulting quantum state (torch.Tensor on CUDA)
    """
    n = int(np.log2(len(quantum_state.reshape(-1, 1)))) # The number of the qubits
    pure_ancilla_list = [] #This is the list of the name of the existing pure_ancilla

    for gate in gate_sequence:
        n_all = n + len(pure_ancilla_list) #The total number of the qubits, including pure_ancilla

        name = gate.get('name')
        parameter = gate.get('parameter', None)
        target = gate.get('target', [])
        control_indices = gate.get('control', [])
        control_sequence = gate.get('control_sequence', [])
        if isinstance(control_sequence, int):
            control_sequence = list(map(int, binary_form_of_integer(control_sequence, len(control_indices))))
        if isinstance(target, int):
            target = [target]
        block_gate_sequence = gate.get('block_gate_sequence', None)

        #Translate target index and control index
        target, control_indices = target_and_control_index_translater(target, control_indices, n, pure_ancilla_list)
        if name == 'create_ancilla':
            quantum_state = create_qubit(quantum_state)
            pure_ancilla_list.append(parameter)
        elif name == 'kill_ancilla':
            quantum_state = kill_qubit(n_all, quantum_state, translate_qubit_index(parameter, n, pure_ancilla_list))
            #quantum_state = kill_nonzero_qubit(quantum_state, translate_qubit_index(parameter, n, pure_ancilla_list))
            pure_ancilla_list.remove(parameter)
        elif name == 'kill_qubit':
            quantum_state =  kill_qubit(n_all, quantum_state, translate_qubit_index(parameter, n, pure_ancilla_list))
            n -= 1
        elif name == 'kill_nonzero_ancilla':
            quantum_state = kill_nonzero_qubit(n_all, quantum_state, translate_qubit_index(parameter, n, pure_ancilla_list))
            pure_ancilla_list.remove(parameter)
        elif name == 'zoom_in':
            quantum_state = zoom_in_permutation(n_all, quantum_state, target, control_indices).reshape([2]*n_all)
            control_extraction = tuple(control_sequence) + (slice(None),) * (n_all - len(control_sequence))
            quantum_state_inside = quantum_state[control_extraction].reshape(-1,1)
            quantum_state_inside = EXECUTE(block_gate_sequence, quantum_state_inside)
            quantum_state[control_extraction]=quantum_state_inside.reshape([2]*(n_all-len(control_sequence)))
            quantum_state = zoom_out_permutation(n_all, quantum_state, target, control_indices)
            # quantum_state = quantum_state.reshape([2 for i in range(n_all)])
            # quantum_state = quantum_state.permute(zoom_in_permutation(n_all, quantum_state, target, control_indices))
            # control_extraction = tuple(control_sequence) + (slice(None),) * (n_all - len(control_sequence))
            # EXECUTE(block_gate_sequence, quantum_state[control_extraction].reshape(-1,1))
            # quantum_state = quantum_state.reshape([2 for i in range(n_all)])
            # quantum_state = quantum_state.permute(zoom_out_permutation(n_all, quantum_state, target, control_indices))
        elif name == 'swap':
            quantum_state = swap(n_all, quantum_state, target[0], target[1])
        else:
            gate_unitary = gate_set[name.lower()] if parameter is None else gate_set[name.lower()](parameter)
            quantum_state = control_one_qubit_rotation(n_all, quantum_state, gate_unitary, control_indices, control_sequence, target[0])
            # control_one_qubit_rotation(n_all, quantum_state, gate_unitary, control_indices, control_sequence, target[0])
    return quantum_state



def translate_qubit_index(qubit, n_register, pure_ancilla_list):
    """
    Converts the qubit to the corresponding integer index. Depending on the type of qubit (integer or string)
    and its specific type identifier, it is mapped to the appropriate index. The qubits ordering: register_qubit, pure_ancilla
    """
    if isinstance(qubit, str):
        return pure_ancilla_list.index(qubit) + n_register
    else:
        return qubit


def target_and_control_index_translater(target, control, n_register, pure_ancilla_list):
    """
    Converts the quantum bit indices in the target and control lists such as pure_ancilla
    to their corresponding integer indices.
    """
    target = [translate_qubit_index(q, n_register, pure_ancilla_list)
              for q in target]
    control = [
        translate_qubit_index(q, n_register, pure_ancilla_list) for q
        in control]
    return target, control



if __name__ == '__main__':
    gate_sequence_small = [
        {"name": "ry", "control_sequence": [0], "control": [1], "target": [0], "parameter": np.pi / 2}]
    gate_sequence = [{"name": "h", 'control_sequence': [], "control": [], "target": [0]}]
    gate_sequence += [{"name": "h", 'control_sequence': [], "control": [], "target": [2]}]
    gate_sequence += [{"name": "x", 'control_sequence': [], "control": [], "target": [3]}]
    gate_sequence += [{"name": "zoom_in", "block_gate_sequence": gate_sequence_small, "target": [1, 0], "control": [2],
                       "control_sequence": [0]}]
    gate_sequence += [{"name": "x", 'control_sequence': [], "control": [], "target": [1]}]
    quantum_state = torch.zeros(2 ** 4, dtype=torch.complex64)
    quantum_state[0] = 1
    print(EXECUTE(gate_sequence, quantum_state))

