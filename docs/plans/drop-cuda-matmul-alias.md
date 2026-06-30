# Drop CUDA Matmul Alias

## Background

CUDA keep work now uses lightweight elementwise ReLU operations. The public
Python API should use the matching `relu_iterations` vocabulary only.

## Goal

Remove the legacy `matmul_iterations` keyword alias from
`CudaGPUController.__init__` so callers get Python's normal unexpected-keyword
`TypeError`.

## Solution

- Replace the old alias validation test with a contract test proving
  `matmul_iterations` is not accepted.
- Remove the alias parameter, docstring text, and override logic from the CUDA
  controller.
- Add a short AGENTS.md guardrail so future CUDA workload tuning keeps the
  `relu_iterations` name unless a documented API decision reopens it.

## Todo

- [x] Add the failing regression test.
- [x] Verify the focused test fails on current code because the alias is still
      accepted.
- [x] Remove the alias from `CudaGPUController`.
- [x] Update the guardrail docs.
- [x] Run targeted tests, docs build, and diff check.

## Verification

- `PYTHONPATH=$PWD/src pytest tests/global_controller/test_contract.py::test_cuda_controller_rejects_matmul_iteration_alias_keyword -q`
- `PYTHONPATH=$PWD/src pytest tests/global_controller/test_contract.py -q`
- `PYTHONPATH=$PWD/src mkdocs build`
- `git diff --check`

RED result: the focused test failed with `Failed: DID NOT RAISE <class
'TypeError'>`, confirming the legacy alias was still accepted.
