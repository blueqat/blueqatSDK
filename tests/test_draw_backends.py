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
