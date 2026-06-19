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

from collections import Counter
import pytest
from blueqat.utils import to_inttuple

@pytest.mark.parametrize('arg, expect', [
    ("01011", (0, 1, 0, 1, 1)),
    ({"00011": 2, "10100": 3}, {(0, 0, 0, 1, 1): 2, (1, 0, 1, 0, 0): 3}),
    (Counter({"00011": 2, "10100": 3}), Counter({(0, 0, 0, 1, 1): 2, (1, 0, 1, 0, 0): 3}))
])
def test_to_inttuple(arg, expect):
    assert to_inttuple(arg) == expect
