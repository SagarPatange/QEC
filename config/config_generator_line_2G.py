"""This module generates JSON config files for networks in a linear configuration.

Help information may also be obtained using the `-h` flag.

Args:
    linear_size (int): number of nodes in the graph.
    memo_size (int): number of memories per node.
    qc_length (float): distance between nodes (in km).
    qc_atten (float): quantum channel attenuation (in dB/m).
    cc_delay (float): classical channel delay (in ms).

Optional Args:
    -d --directory (str): name of the output directory (default tmp)
    -o --output (str): name of the output file (default out.json).
    -s --stop (float): simulation stop time (in s) (default infinity).
    -n --nodes (str): path to csv file providing process information for nodes.
"""

# For generating a 2-node linear network with 20 memories per node, 1 km distance between nodes, for 1G networks
# python config/config_generator_line.py 2 20 1 0.0002 1 -d config -o line_2_physical.json -s 10


# For 2nd generation quantum routers (NEW):
# python config/config_generator_line_2G.py 3 7 1 0.0002 -1 -d config -o line_3_2G.json -s 10 --gen2 -ds 7

import argparse
import json
import os
from config_generator import add_default_args, generate_nodes, final_config, router_name_func
from sequence.topology.topology import Topology
from sequence.topology.router_net_topo import RouterNetTopo


#=========================== NEW FUNCTION ============================
def generate_2g_nodes(router_names, memo_size, data_memo_size, ancilla_memo_size, template=None, gate_fidelity=None, two_qubit_gate_fidelity=None, ft_prep_mode="none", ft_max_retries=3, idle_t1_sec=1e12, idle_t2_sec=1e12, idle_pauli_weights=None):
    """Generate node configs for 2nd generation quantum routers."""
    # Start with standard nodes
    nodes = generate_nodes(router_names, memo_size, template)

    N = len(router_names)

    for i, node in enumerate(nodes):
        node["generation"] = 2           # This triggers 2nd gen creation
        # FT ancilla-prep configuration (consumed by app/router layers later)
        node["ft_prep_mode"] = ft_prep_mode
        node["ft_max_retries"] = int(ft_max_retries)

        if i == 0 or i == N-1:
            node["data_memo_size"] = data_memo_size
            node["ancilla_memo_size"] = ancilla_memo_size
        else:
            node["memo_size"] = memo_size*2
            node["data_memo_size"] = data_memo_size*2
            node["ancilla_memo_size"] = ancilla_memo_size*2

        if gate_fidelity is not None:
            node["gate_fidelity"] = gate_fidelity
        if two_qubit_gate_fidelity is not None:
            node["two_qubit_gate_fidelity"] = two_qubit_gate_fidelity
        node["idle_t1_sec"] = float(idle_t1_sec)
        node["idle_t2_sec"] = float(idle_t2_sec)
        if idle_pauli_weights is None:
            node["idle_pauli_weights"] = {"x": 0.05, "y": 0.05, "z": 0.90}
        else:
            node["idle_pauli_weights"] = {
                "x": float(idle_pauli_weights["x"]),
                "y": float(idle_pauli_weights["y"]),
                "z": float(idle_pauli_weights["z"]),
            }

    return nodes
#================================================================


parser = argparse.ArgumentParser()
parser.add_argument('linear_size', type=int, help='number of network nodes')
parser = add_default_args(parser)
parser.add_argument('-ds', '--data_size', type = int, default = 7, help='Data memories for 2nd generation quantum routers')
parser.add_argument('-as', '--ancilla_size', type = int, default = 0, help='Ancilla memories for 2nd generation quantum routers')
parser.add_argument('--gate_fid', type=float, default=None, help='Single-qubit gate fidelity (default: ideal)')
parser.add_argument('--two_qubit_gate_fid', type=float, default=None, help='Two-qubit gate fidelity (default: ideal)')
parser.add_argument('--css_code', type=str, default='[[7,1,3]]', help='CSS code name, e.g. "[[7,1,3]]" or "[[9,1,3]]". Auto-sets data_size.')
parser.add_argument('--ft_prep_mode', type=str, default='none', choices=['none', 'minimal', 'strong'], help='Fault-tolerant prep mode for logical-state preparation')
parser.add_argument('--ft_max_retries', type=int, default=3, help='Max retries for FT prep attempts per logical block')
parser.add_argument('--idle_t1_sec', type=float, default=3e12, help='Shared idling T1 time in seconds')
parser.add_argument('--idle_t2_sec', type=float, default=3e12, help='Shared idling T2 time in seconds')
parser.add_argument('--idle_pauli_x', type=float, default=0.05, help='Idle Pauli X weight')
parser.add_argument('--idle_pauli_y', type=float, default=0.05, help='Idle Pauli Y weight')
parser.add_argument('--idle_pauli_z', type=float, default=0.90, help='Idle Pauli Z weight')

#=========================== Changes ============================
parser.add_argument('--gen2', action='store_true', help='Use 2nd generation quantum routers')
#================================================================

args = parser.parse_args()

# If --css_code is provided, derive data_size and memo_size from the code's n
if args.css_code != '[[7,1,3]]' or args.data_size == 7:
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from css_codes import get_css_code
    code = get_css_code(args.css_code)
    args.data_size = code.n
    args.memo_size = code.n

output_dict = {}

# templates
output_dict[Topology.ALL_TEMPLATES] = \
    {
        "qec": {
            "memory": {
                "fidelity": 1.0,
                "efficiency": 0.9
            }
        },
    }

# get csv file (if present)
router_names = [router_name_func(i) for i in range(args.linear_size)]
nodes = generate_nodes(router_names, args.memo_size)
template = 'qec'

#=========================== MODIFIED SECTION ============================

if args.gen2:  # Check for 2nd generation flag (NEW)
    args.formalism = 'tableau'
    idle_pauli_weights = {"x": args.idle_pauli_x, "y": args.idle_pauli_y, "z": args.idle_pauli_z}
    nodes = generate_2g_nodes(router_names, args.memo_size, args.data_size, args.ancilla_size,
                              template, args.gate_fid, args.two_qubit_gate_fid,
                              args.ft_prep_mode, args.ft_max_retries,
                              args.idle_t1_sec, args.idle_t2_sec, idle_pauli_weights)
else:
    nodes = generate_nodes(router_names, args.memo_size, template)
#================================================================


# generate bsm nodes
bsm_names = ["BSM_{}_{}".format(i, i + 1) for i in range(args.linear_size - 1)]
bsm_nodes = [{Topology.NAME: bsm_name,
              Topology.TYPE: RouterNetTopo.BSM_NODE,
              Topology.SEED: i,
              RouterNetTopo.TEMPLATE: template}
             for i, bsm_name in enumerate(bsm_names)]
nodes += bsm_nodes
output_dict[Topology.ALL_NODE] = nodes

# generate quantum links, classical with bsm nodes
qchannels = []
cchannels = []

if args.cc_delay < 0:
    cc_delay = -1
else:   
    cc_delay = int(args.cc_delay * 1e9)
    
for i, bsm_name in enumerate(bsm_names):
    # qchannels
    qchannels.append({Topology.SRC: router_names[i],
                      Topology.DST: bsm_name,
                      Topology.DISTANCE: args.qc_length * 1000 / 2,
                      Topology.ATTENUATION: args.qc_atten})
    qchannels.append({Topology.SRC: router_names[i + 1],
                      Topology.DST: bsm_name,
                      Topology.DISTANCE: args.qc_length * 1000 / 2,
                      Topology.ATTENUATION: args.qc_atten})
    # cchannels
    for node in [router_names[i], router_names[i + 1]]:
        cchannels.append({Topology.SRC: bsm_name,
                          Topology.DST: node,
                          Topology.DISTANCE: args.qc_length * 1000 / 2,
                          Topology.DELAY: cc_delay})
        cchannels.append({Topology.SRC: node,
                          Topology.DST: bsm_name,
                          Topology.DISTANCE: args.qc_length * 1000 / 2,
                          Topology.DELAY: cc_delay})
        
output_dict[Topology.ALL_Q_CHANNEL] = qchannels

# generate router-to-router classical links
# NOTE: Cannot use SeQUeNCe's generate_classical(router_names, cc_delay) because
# it multiplies cc_delay by 1e9 again (line 144 of config_generator.py), but our
# cc_delay is already -1 (sentinel for "auto-calculate"). The double multiplication
# produces -1e9, which bypasses the sentinel check in ClassicalChannel.__init__
# (it only checks delay == -1) and results in a literal negative delay.
# We generate these manually with the correct cc_delay value.
for n1 in router_names:
    for n2 in router_names:
        if n1 != n2:
            cchannels.append({Topology.SRC: n1,
                              Topology.DST: n2,
                              Topology.DELAY: cc_delay})
output_dict[Topology.ALL_C_CHANNEL] = cchannels


# write other config options to output dictionary
final_config(output_dict, args)

# write final json
path = os.path.join(args.directory, args.output)
output_file = open(path, 'w')
json.dump(output_dict, output_file, indent=4)
