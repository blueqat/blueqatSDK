"""Every module in the blueqat package must be importable.

Guards against two real regressions seen in the wild:
- a subpackage missing from pyproject's package list (2.1.0: blueqat.eo
  absent from non-editable installs) -- caught here only when the suite runs
  against a real install, as CI's non-editable job does;
- an unquoted annotation referencing a name that was never imported
  (2.1.1: `Sequence` in blueqat.eo.optimizer). Python <= 3.13 evaluates
  annotations at import time, so importing every module catches it there;
  Python >= 3.14 defers them (PEP 649), so we also force evaluation below.
"""
import importlib
import inspect
import pkgutil

import blueqat


def _walk_modules():
    yield 'blueqat'
    for m in pkgutil.walk_packages(blueqat.__path__, prefix='blueqat.'):
        yield m.name


def test_all_modules_import():
    for name in _walk_modules():
        importlib.import_module(name)


def test_all_annotations_resolve():
    # Force evaluation of (lazy) unquoted annotations on every function and
    # class, so missing-import bugs surface even on Python >= 3.14.
    for name in _walk_modules():
        mod = importlib.import_module(name)
        for objname, obj in vars(mod).items():
            if getattr(obj, '__module__', None) != name:
                continue
            targets = [obj] if inspect.isfunction(obj) else []
            if inspect.isclass(obj):
                targets.append(obj)
                for meth in vars(obj).values():
                    f = meth.__func__ if isinstance(
                        meth, (classmethod, staticmethod)) else meth
                    if inspect.isfunction(f):
                        targets.append(f)
            for t in targets:
                inspect.get_annotations(t)  # raises NameError on missing names
