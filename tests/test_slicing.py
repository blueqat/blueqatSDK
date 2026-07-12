# Copyright 2019 The Blueqat Developers
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

from itertools import chain
import pytest
from blueqat.gate import slicing

class _GetItem:
    def __getitem__(self, args):
        return args

s = _GetItem()

def assert_slicing(actual, expected):
    actual = list(actual)
    expected = list(expected)
    assert actual == expected

def test_subscription1():
    assert_slicing(slicing(0, 10), [0])

def test_subscription2():
    assert_slicing(slicing(-1, 10), [9])

def test_subscription3():
    assert_slicing(slicing(-2, 10), [8])

def test_subscription4():
    assert_slicing(slicing(9, 10), [9])

def test_subscription5():
    with pytest.raises(TypeError):
        list(slicing(1.2, 10))

def test_multi_subscription1():
    assert_slicing(slicing((1, -1), 10), [1, 9])

def test_multi_subscription2():
    assert_slicing(slicing((4, 2, 1, -2, 5), 10), [4, 2, 1, 8, 5])

def test_multi_subscription3():
    it = slicing((0, 1, 2.3, 4, 5), 10)
    assert next(it) == 0
    assert next(it) == 1
    with pytest.raises(TypeError):
        next(it)
    # assert next(it) == 4
    # assert next(it) == 5
    with pytest.raises(StopIteration):
        next(it)
    with pytest.raises(StopIteration):
        next(it)

def test_slicing1():
    assert_slicing(slicing(s[1:3], 10), range(10)[1:3])

def test_slicing2():
    assert_slicing(slicing(s[:3], 10), range(10)[:3])

def test_slicing3():
    assert_slicing(slicing(s[:], 4), range(4)[:])

def test_slicing4():
    assert_slicing(slicing(s[2:], 4), range(4)[2:])

def test_slicing5():
    assert_slicing(slicing(s[::2], 10), range(10)[::2])

def test_slicing6():
    assert_slicing(slicing(s[1::2], 10), range(10)[1::2])

def test_slicing7():
    assert_slicing(slicing(s[100::2], 10), range(10)[100::2])

def test_slicing8():
    assert_slicing(slicing(s[1:10:3], 10), range(10)[1:10:3])

def test_slicing9():
    assert_slicing(slicing(s[10:0], 10), range(10)[10:0])

def test_slicing10():
    assert_slicing(slicing(s[8:3], 10), range(10)[8:3])

def test_slicing11():
    assert_slicing(slicing(s[::-1], 10), range(10)[::-1])

def test_slicing12():
    assert_slicing(slicing(s[::-3], 10), range(10)[::-3])

def test_slicing13():
    assert_slicing(slicing(s[8::-2], 10), range(10)[8::-2])

def test_slicing14():
    assert_slicing(slicing(s[7::-2], 10), range(10)[7::-2])

def test_slicing15():
    assert_slicing(slicing(s[:3:-1], 10), range(10)[:3:-1])

def test_slicing16():
    assert_slicing(slicing(s[:4:-1], 10), range(10)[:4:-1])

def test_slicing17():
    assert_slicing(slicing(s[8:4:-1], 10), range(10)[8:4:-1])

def test_multi_slicing1():
    assert_slicing(slicing(s[2:4:8, 3, 8:2:-1], 10), chain(range(10)[2:4:8], [range(10)[3]], range(10)[8:2:-1]))
