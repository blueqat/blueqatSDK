import matplotlib
matplotlib.use('Agg')  # headless: avoid blocking on an interactive plt.show()

from blueqat import Circuit


def test_draw_circuit_backend():
    result = Circuit().h[0].cx[0, 1].m[:].run(backend='draw')
    # DrawCircuit returns the raw layout data alongside layout bookkeeping.
    assert isinstance(result, tuple)
    qlist = result[0]
    assert 0 in qlist and 1 in qlist


def test_draw_tn_backend_structure():
    graph = Circuit().h[0].cx[0, 1].h[1].run(backend='draw_tn', show=False)
    # 2 initial-qubit tensors + h + cx + h = 5 real tensor nodes, plus 2 open legs.
    real_nodes = [n for n in graph.nodes if not graph.nodes[n].get('open_leg')]
    open_nodes = [n for n in graph.nodes if graph.nodes[n].get('open_leg')]
    assert len(real_nodes) == 5
    assert len(open_nodes) == 2
    assert graph.number_of_edges() == 6


def test_draw_tn_backend_show_renders():
    # Should not raise even with show=True (the default), using the Agg backend.
    Circuit().h[0].cx[0, 1].run(backend='draw_tn')


def test_draw_supports_every_gate_in_gateset():
    """Every gate in GATE_SET must be drawable: none may be silently omitted
    from the diagram (regression test for 14 gates -- cphase, u, crx, ... --
    that the drawer used to drop without any indication)."""
    import warnings

    import matplotlib.pyplot as plt
    import torch

    from blueqat.gateset import GATE_SET

    ONE_PARAM_1Q = {'p', 'phase', 'r', 'rx', 'ry', 'rz'}
    ONE_PARAM_2Q = {'cp', 'cphase', 'cr', 'crx', 'cry', 'crz',
                    'rxx', 'ryy', 'rzz', 'exch', 'exchange'}
    NO_PARAM_2Q = {'cnot', 'cx', 'cy', 'cz', 'ch', 'swap', 'zz', 'zzdg',
                   'iswap', 'iswapdg'}
    THREE_Q = {'ccx', 'toffoli', 'ccz', 'cswap'}

    hadamard = torch.tensor([[1, 1], [1, -1]],
                            dtype=torch.complex128) / (2 ** 0.5)

    for name in GATE_SET:
        c = Circuit(3)
        wrapper = getattr(c, name)
        if name == 'mat1':
            wrapper(hadamard)[0]
        elif name == 'u':
            wrapper(0.1, 0.2, 0.3)[0]
        elif name == 'cu':
            wrapper(0.1, 0.2, 0.3, 0.4)[0, 1]
        elif name in ONE_PARAM_1Q:
            wrapper(0.5)[0]
        elif name in ONE_PARAM_2Q:
            wrapper(0.5)[0, 1]
        elif name in NO_PARAM_2Q:
            wrapper[0, 1]
        elif name in THREE_Q:
            wrapper[0, 1, 2]
        else:  # 1-qubit no-param gates, measure, reset, barrier
            wrapper[0]

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            c.run(backend='draw')
        plt.close('all')
        omitted = [w for w in caught
                   if 'not supported by the draw backend' in str(w.message)]
        assert not omitted, f"gate '{name}' is not drawable: {omitted[0].message}"
