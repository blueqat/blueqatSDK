from blueqat import Circuit

def test_key0():
    assert Circuit().m(key="test")[0].run(shots=10, returns="samples") == [{"test": [0]} for _ in range(10)]


def test_key1():
    assert Circuit().x[0].m(key="test")[0].run(shots=10, returns="samples") == [{"test": [1]} for _ in range(10)]


def test_keys():
    assert Circuit().x[0].m(key="a")[0, 1].x[0].m(key="b")[0].run(shots=10, returns="samples") == [{"a": [1, 0], "b": [0]} for _ in range(10)]


def test_key_replace():
    assert Circuit().x[0].m(key="a")[0, 1].x[0].m(key="a", duplicated="replace")[0].run(shots=10, returns="samples") == [{"a": [0]} for _ in range(10)]


def test_key_append():
    assert Circuit().x[0].m(key="a")[0, 1].x[0].m(key="a", duplicated="append")[0].run(shots=10, returns="samples") == [{"a": [1, 0, 0]} for _ in range(10)]
