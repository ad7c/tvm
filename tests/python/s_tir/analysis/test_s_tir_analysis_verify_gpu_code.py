# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""Tests for detailed GPU code verification errors."""

import tvm.testing
from tvm.s_tir import analysis
from tvm.script import tirx as T


@T.prim_func(s_tir=True)
def ValidThreads():
    thread_idx_x = T.env_thread("threadIdx.x")
    T.launch_thread(thread_idx_x, 32)
    T.evaluate(0)


@T.prim_func(s_tir=True)
def TooManyThreads():
    thread_idx_x = T.env_thread("threadIdx.x")
    T.launch_thread(thread_idx_x, 2048)
    T.evaluate(0)


CONSTRAINTS = {
    "max_shared_memory_per_block": 49152,
    "max_threads_per_block": 1024,
    "max_vthread": 8,
    "max_vector_bytes": 16,
}


def test_verify_gpu_code_errors_empty_on_success():
    assert analysis.verify_gpu_code(ValidThreads, CONSTRAINTS)
    assert analysis.verify_gpu_code_errors(ValidThreads, CONSTRAINTS) == []


def test_verify_gpu_code_errors_preserve_detail():
    errors = analysis.verify_gpu_code_errors(TooManyThreads, CONSTRAINTS)
    assert not analysis.verify_gpu_code(TooManyThreads, CONSTRAINTS)
    assert len(errors) == 1
    assert "threads per block" in errors[0]
    assert "2048" in errors[0]
    assert "1024" in errors[0]


if __name__ == "__main__":
    tvm.testing.main()
