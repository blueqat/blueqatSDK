from blueqat import Circuit, ParametrizedCircuit

def compare_circuit(c1: Circuit, c2: Circuit) -> bool:
    return repr(c1) == repr(c2)

def test_parametrized1():
    assert compare_circuit(
        ParametrizedCircuit().ry('a')[0].rz('b')[0].subs([1.2, 3.4]),
        Circuit().ry(1.2)[0].rz(3.4)[0])


def test_parametrized2():
    assert compare_circuit(
        ParametrizedCircuit().ry('a')[0].rz('b')[0].subs({'a': 1.2, 'b': 3.4}),
        Circuit().ry(1.2)[0].rz(3.4)[0])


def test_parametrized3():
    assert compare_circuit(
        ParametrizedCircuit().subs([]),
        Circuit()
    )
