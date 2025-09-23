from venv import create
import copy
import time
import numpy as np
import torch
from config_manager import DeviceManager, DatetypeManager
from fundamental_gates_unitary import gate_names

#To make sure the functions is executed in GPU or CPU
device_manager = DeviceManager()
device = device_manager.get_device()
# Set the datetype
datetype_manager = DatetypeManager()
datetype = datetype_manager.get_datetype()

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
        # if torch.abs(gate_unitary[0, 0]) >= 1 / 2:
        #     quantum_state[tuple(control_sequence)][0] = gate_unitary[0,0] * quantum_state[tuple(control_sequence)][0] + gate_unitary[0,1] * quantum_state[tuple(control_sequence)][1]
        #     quantum_state[tuple(control_sequence)][1] = (gate_unitary[1, 0] / gate_unitary[0, 0] * quantum_state[tuple(control_sequence)][0]
        #                                                  + (gate_unitary[1, 1] - gate_unitary[1, 0] / gate_unitary[0, 0] * gate_unitary[0, 1]) * quantum_state[tuple(control_sequence)][1])
        # else:
        #     quantum_state[tuple(control_sequence)] = quantum_state[tuple(control_sequence)].flip(0)
        #     quantum_state[tuple(control_sequence)][0] = gate_unitary[0, 0] * quantum_state[tuple(control_sequence)][1] + gate_unitary[0, 1] * quantum_state[tuple(control_sequence)][0]
        #     quantum_state[tuple(control_sequence)][1] =  (gate_unitary[1, 1] / gate_unitary[0, 1] * quantum_state[tuple(control_sequence)][0] +
        #                                                   (gate_unitary[1, 0] - gate_unitary[1, 1] / gate_unitary[0, 1] * gate_unitary[0, 0]) * quantum_state[tuple(control_sequence)][1])
        #permute back
        quantum_state = quantum_state.permute(inverse_permutation)
        return quantum_state
    return one_qubit_rotation(n, quantum_state, gate_unitary, target_index)


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
        # transposed_tensor = transposed_tensor.reshape(-1)
        # target_tensor0 = gate_unitary[0,0] * transposed_tensor[:2**(n-1)] + gate_unitary[0,1] * transposed_tensor[2**(n-1):2**n]
        # target_tensor1 = gate_unitary[1,0] * transposed_tensor[:2**(n-1)] + gate_unitary[1,1] * transposed_tensor[2**(n-1):2**n]
        # aux_tensor = torch.stack([target_tensor0, target_tensor1], dim=0)
        # aux_tensor = aux_tensor.reshape([2 for i in range(n)])
        # result_tensor = aux_tensor.transpose(0, target_index)
    else:
        result_tensor = gate_unitary @ quantum_state
    return result_tensor
    # if n > 1:
    #     quantum_state = quantum_state.reshape([2 for i in range(n)])
    #     quantum_state = quantum_state.transpose(0, target_index)
    #     if torch.abs(gate_unitary[0,0]) >= 1 / 2:
    #         quantum_state[0] = gate_unitary[0, 0] * quantum_state[0] + gate_unitary[0, 1] * quantum_state[1]
    #         quantum_state[1] = gate_unitary[1, 0] / gate_unitary[0, 0] * quantum_state[0] + (gate_unitary[1, 1] - gate_unitary[1, 0] / gate_unitary[0, 0] * gate_unitary[0, 1]) * quantum_state[1]
    #     else:
    #         quantum_state = quantum_state.flip(0)
    #         quantum_state[0] = gate_unitary[0, 0] * quantum_state[1] + gate_unitary[0, 1] * quantum_state[0]
    #         quantum_state[1] = gate_unitary[1, 1] / gate_unitary[0, 1] * quantum_state[0] + (gate_unitary[1, 0] - gate_unitary[1, 1] / gate_unitary[0, 1] * gate_unitary[0, 0]) * quantum_state[1]
    #     quantum_state = quantum_state.transpose(0, target_index)
    # else:
    #     quantum_state = gate_unitary @ quantum_state
    # return quantum_state


def one_qubit_rotation_broadcast(n, quantum_state, gate_unitary, target_index):
    """
    :param n: the number of qubits
    :param quantum_state: input quantum state, it should be a 2**n normalized torch vector
    :param gate_unitary: one-qubit gate in unitary matrix form, it should be a 2*2 unitary torch matrix
    :param target_index: the index of the target qubit, it should be an int number
    :return: output quantum state
    """
    quantum_state = quantum_state.reshape([2 for i in range(n)])
    quantum_state = quantum_state.transpose(n - 2, target_index)
    quantum_state = gate_unitary @ quantum_state
    quantum_state = quantum_state.transpose(n - 2, target_index)
    return quantum_state


def swap(n, quantum_state, target_index1, target_index2):
    """
    This function swap two quantum qubits
    :param n: the number of qubits
    :param quantum_state: input quantum state, it should be a 2**n normalized torch vector
    :param target_index1: target index 1
    :param target_index2: target index 2
    :return: out put quantum state
    """
    #n = int(np.log2(len(quantum_state.reshape(-1, 1))))
    return quantum_state.reshape([2 for i in range(n)]).transpose(target_index1,target_index2)

def create_qubit(quantum_state):
    """
    This function create a new qubit with state |0> and put it at the end of all the qubits
    :param quantum_state: input quantum state, it should be a 2**n normalized torch vector
    :return: output quantum state
    """
    return torch.kron(quantum_state,torch.tensor([1,0], dtype=datetype, device=device))

def create_name_for_ancilla():
    """
    This function create names for the pure ancilla
    :return: A list containing the names of the pure ancilla
    """
    name=str(time.time())
    time.sleep(0.000001)#wait 1 mcs
    return name

def kill_qubit(n, quantum_state, index):
    """
    This function kill the qubit whose index is 'index', note that The qubit is only allowed to be killed when it is in the |0> state
    :param n: the number of qubits
    :param quantum_state: input quantum state, it should be a 2**n normalized torch vector
    :param index: the index of the qubit which should be killed.
    :return: output quantum state
    """
    #n = int(np.log2(len(quantum_state.reshape(-1,1))))
    tens = quantum_state.reshape([2 for i in range(n)])
    #if  sum(tens.permute(tuple([index] + list(range(index)) + list(range(index + 1, n))))[1].reshape(-1)) != 0:
        #raise ValueError("The qubit is not in the |0> state, the kill_qubit operation cannot be performed.")
    return tens.permute(tuple([index] + list(range(index)) + list(range(index + 1, n))))[0]

def kill_nonzero_qubit(n, quantum_state, indices):
    """
    This function kill the qubit whose index is 'index', note that this function can kill qubits the state of which is not |0>
    :param n: the number of qubits
    :param quantum_state: input quantum state, it should be a 2**n normalized torch vector
    :param indices: the indices of the qubits which should be killed.
    :return: output quantum state
    """
    if isinstance(indices, int):
        indices = [indices]
    #n = int(np.log2(len(quantum_state.reshape(-1,1))))
    if len(indices) == n:
        raise ValueError("The number of the qubits been killed should be less than the total number of qubits.")
    tens = quantum_state.reshape([2 for _ in range(n)])
    temp = tens.permute(tuple(indices + [x for x in range(n) if x not in indices]))[tuple([0] * len(indices))]
    return temp/torch.sqrt(torch.sum(temp ** 2))



def zoom_in_permutation(n, quantum_state, target_indices, control_indices):
    """
    This function acts zoom_in operation on gate_sequence
    :param n: the number of qubits
    :param quantum_state: input quantum state, it should be a 2**n normalized torch vector
    :param target_indices:
    :param control_indices:
    :return: output quantum state
    """
    permutation = tuple(control_indices + target_indices + [i for i in range(n) if i not in control_indices + target_indices])
    return quantum_state.reshape([2 for i in range(n)]).permute(permutation).reshape(-1,1)
    # return permutation

def zoom_out_permutation(n, quantum_state, target_indices, control_indices):
    """
    This function acts zoom_out operation on gate_sequence
    :param n: the number of qubits
    :param quantum_state: input quantum state, it should be a 2**n normalized torch vector
    :param target_indices:
    :param control_indices:
    :return: output quantum state
    """
    permutation = control_indices + target_indices + [i for i in range(n) if i not in control_indices + target_indices]
    inverse_permutation = tuple(np.argsort(np.array(permutation)))
    return quantum_state.reshape([2 for i in range(n)]).permute(inverse_permutation).reshape(-1,1)
    # return inverse_permutation


def ancilla_expand(initial, n, k, num):#k is position index
    tens = initial.reshape([2 for i in range(n)])
    new_dim = [2 for i in range(n + num)]
    expanded = torch.zeros(new_dim)
    expanded[(slice(None),) * k + (0,) *  num + (slice(None),) * (n-k)] = tens.clone()
    return expanded


def ancilla_kill_old_version(initial, n, k, num):
    tens = initial.reshape([2 for i in range(n)])
    return tens[(slice(None),) * k + (0,) * num + (slice(None),) * (n-k-num)]


# def dagger(gate_sequence):
#     gate_sequence_dagger = copy.deepcopy(gate_sequence[::-1])
#     for i in range(len(gate_sequence_dagger)):
#         gate = gate_sequence_dagger[i]
#         name = gate.get('name')
#         parameter = gate.get('parameter', None)
#         block_gate_sequence = gate.get('block_gate_sequence', None)
#         if name.lower() == 'ry' or name.lower() == 'rz' or name.lower() == 'rx' or name.lower() == 'phase_gate' or name.lower() == 'global_phase':
#             gate_sequence_dagger[i]['parameter'] = - parameter
#         if name.lower() == 's':
#             gate_sequence_dagger[i]['name'] = 's_dagger'
#         if name.lower() == 'u':
#             temp = copy.deepcopy(gate_sequence_dagger[i]['parameter'])
#             gate_sequence_dagger[i]['parameter'][0] = -temp[0]
#             gate_sequence_dagger[i]['parameter'][1] = -temp[2]
#             gate_sequence_dagger[i]['parameter'][2] = -temp[1]
#         if name == 'kill_ancilla':
#             gate_sequence_dagger[i]['name'] = 'create_ancilla'
#         if name == 'create_ancilla':
#             gate_sequence_dagger[i]['name'] = 'kill_ancilla'
#         if name == 'zoom_in':
#             gate_sequence_dagger[i]['block_gate_sequence'] = dagger(block_gate_sequence)
#     return gate_sequence_dagger


def dagger(gate_sequence):
    gate_sequence_dagger = []
    for gate in reversed(gate_sequence):
        name = gate.get('name')
        parameter = gate.get('parameter', None)
        target = gate.get('target', [])
        control_indices = gate.get('control', [])
        control_sequence = gate.get('control_sequence', [])
        block_gate_sequence = gate.get('block_gate_sequence', None)

        dagger_gate = {}
        # renew name
        if name == 'create_ancilla':
            dagger_gate['name'] = 'kill_ancilla'
        elif name == 'kill_ancilla':
            dagger_gate['name'] = 'create_ancilla'
        elif name == 's':
            dagger_gate['name'] = 's_dagger'
        elif name == 's_dagger':
            dagger_gate['name'] = 's'
        else:
            dagger_gate['name'] = name
        # renew parameter
        if parameter is not None:
            if name.lower() == 'ry' or name.lower() == 'rz' or name.lower() == 'rx' or name.lower() == 'phase_gate' or name.lower() == 'global_phase':
                dagger_gate['parameter'] = -parameter
            elif name.lower() == 'u':
                dagger_gate['parameter'] = [-parameter[0], -parameter[2], -parameter[1]]
            else:
                dagger_gate['parameter'] = parameter
        # renew target
        if target is not []:
            dagger_gate['target'] = target
        # renew control
        if control_indices is not []:
            dagger_gate['control'] = control_indices
        # renew control_sequence
        if control_sequence is not []:
            dagger_gate['control_sequence'] = control_sequence
        # renew block_gate_sequence
        if block_gate_sequence is not None:
            dagger_gate['block_gate_sequence'] = dagger(block_gate_sequence)

        gate_sequence_dagger.append(dagger_gate)
    return gate_sequence_dagger

def create_quantum_state(n, i):
    """
    This function creates a quantum state with n qubits in state |i>
    :param n: The number of qubits
    :param i: The quantum state |i>
    :return: output quantum state
    """
    quantum_state = torch.zeros(2 ** n, dtype=datetype, device=device)
    quantum_state[i] = torch.tensor(1, dtype=datetype, device=device)
    return quantum_state

def gate_number_counting(gate_sequence):
    one_qubit_gate = 0
    control_one_qubit_gate = 0
    for gate in gate_sequence:
        name = gate.get('name')
        control_indices = gate.get('control', [])
        block_gate_sequence = gate.get('block_gate_sequence', None)
        if name == 'zoom_in':
            one_qubit_gate_inside, control_one_qubit_gate_inside = gate_number_counting(block_gate_sequence)
            if len(control_indices) != 0:
                control_one_qubit_gate += one_qubit_gate_inside
                control_one_qubit_gate += control_one_qubit_gate_inside
            else:
                one_qubit_gate += one_qubit_gate_inside
                control_one_qubit_gate += control_one_qubit_gate_inside
        elif name.lower() in gate_names:
            if len(control_indices) != 0:
                control_one_qubit_gate += 1
            else:
                one_qubit_gate += 1
    return one_qubit_gate, control_one_qubit_gate
