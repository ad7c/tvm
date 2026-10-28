# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements. See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership. The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License. You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied. See the License for the
# specific language governing permissions and limitations
# under the License.
"""Opt-in NumPy/CUDA experiment environment, inherited by Popen workers.

TVM-FFI eagerly imports an optional PyTorch installation on Windows. These
experiments do not use PyTorch, and loading its DLLs in each compiler worker
can exhaust the system commit limit. Treat it as an unavailable optional
dependency in this profile; do not install a fake module or modify packages.
"""

import os
import sys


if os.environ.get("TVM_EXPERIMENT_NO_TORCH") == "1":
    class _ExcludeExperimentTorch:
        def find_spec(self, fullname, path=None, target=None):
            if fullname == "torch" or fullname.startswith("torch."):
                raise ModuleNotFoundError(
                    "PyTorch is excluded by TVM_EXPERIMENT_NO_TORCH", name=fullname
                )
            return None

    sys.meta_path.insert(0, _ExcludeExperimentTorch())
