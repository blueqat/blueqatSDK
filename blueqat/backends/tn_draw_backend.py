# Copyright 2019-2026 The Blueqat Developers
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Backend for visualizing the tensor-network graph structure (nodes = tensors,
edges = contracted/shared indices) that TorchBackend's tensornet mode builds.
"""

from typing import Any, Dict, List, Optional

import torch

from ..gate import Operation
from .backendbase import Backend
from .torch_backend import TorchBackend, TorchBackendContext


class TNGraphDrawBackend(Backend):
    """Backend which draws the tensor-network contraction graph for a circuit,
    instead of running it. Each gate/initial-qubit tensor is a node; an edge
    connects two tensors that share a contracted axis. Still-open (uncontracted)
    qubit legs are drawn as dangling leaf nodes.
    """

    def run(self, gates: List[Operation], n_qubits: int, *args: Any, **kwargs: Any) -> Any:
        import matplotlib.pyplot as plt
        import networkx as nx

        device = kwargs.get('device', torch.device('cpu'))
        dtype = kwargs.get('dtype', torch.complex128)
        show = kwargs.get('show', True)

        tn_backend = TorchBackend(mode='tensornet', device=device, dtype=dtype)
        ctx = TorchBackendContext(n_qubits, 'tensornet', device, dtype)
        ctx = tn_backend._run_inner(ctx, gates, n_qubits)

        graph = nx.MultiGraph()
        for i, tensor in enumerate(ctx.tensors):
            graph.add_node(i, shape=tuple(tensor.shape))

        axis_to_nodes: Dict[int, List[int]] = {}
        for i, idxs in enumerate(ctx.tensor_indices):
            for axis in idxs:
                axis_to_nodes.setdefault(axis, []).append(i)

        for axis, nodes in axis_to_nodes.items():
            if len(nodes) == 2:
                graph.add_edge(nodes[0], nodes[1], label=str(axis))
            elif len(nodes) == 1:
                leaf = f'open_{axis}'
                graph.add_node(leaf, shape=None, open_leg=True)
                graph.add_edge(nodes[0], leaf, label=f'q{axis}')
            # a well-formed tensor network never shares one axis across more
            # than 2 tensors, so any other case is left undrawn defensively.

        if show:
            pos = nx.spring_layout(graph, seed=0)
            node_colors = ['white' if graph.nodes[n].get('open_leg') else '#0BB0E2' for n in graph.nodes]
            nx.draw_networkx(graph, pos, node_color=node_colors, with_labels=True)
            plt.show()

        return graph
