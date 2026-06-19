import pytest
import numpy as np

from blueqat import Circuit, circuitmacro


@circuitmacro
def xh(c):
    return c.x[0].h[1]


@circuitmacro()
def hh(c):
    return c.h[0, 1]


@circuitmacro
def ryrz(c, p, q):
    return c.ry(p)[0].rz(q)[0]


@circuitmacro('pizza')
def pasta(c):
    return c.x[0]


def test_macro1():
    assert np.allclose(Circuit().x[0].h[1].run(), Circuit().xh().run())


def test_macro2():
    assert np.allclose(Circuit().h[0, 1].run(), Circuit().hh().run())


def test_macro3():
    assert np.allclose(Circuit().x[0].run(), Circuit().pizza().run())


def test_macro4():
    with pytest.raises(AttributeError):
        Circuit().pasta()


def test_macro5():
    assert np.allclose(Circuit().ry(0.3)[0].rz(0.5)[0].run(),
                       Circuit().ryrz(0.3, 0.5).run())


def test_macro6():
    with pytest.raises(ValueError):
        @circuitmacro(allow_overwrite=False)
        def ryrz(c, p):
            return c.ry(p)[0]


def test_macro7():
    with pytest.raises(ValueError):
        @circuitmacro(allow_overwrite=False)
        def run(c):
            return c


def test_macro8():
    with pytest.warns(UserWarning):
        @circuitmacro
        def run(c):
            return c


def test_macro9():
    with pytest.raises(ValueError):
        @circuitmacro(allow_overwrite=False)
        def xh(c):
            return c
    assert np.allclose(Circuit().x[0].h[1].run(), Circuit().xh().run())


def test_macro10():
    @circuitmacro
    def xh(c):
        return Circuit().x[1].h[0]
    assert np.allclose(Circuit().x[1].h[0].run(), Circuit().xh().run())
    @circuitmacro
    def xh(c):
        return Circuit().x[0].h[1]
