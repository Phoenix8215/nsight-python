.. SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: Apache-2.0

Quickstart
-------------

Here's the absolute minimal example to get started with Nsight Python. Just add a decorator to your function and wrap the kernel you want to profile with ``nsight.annotate()``:

.. code-block:: python

   import torch
   import nsight

   @nsight.analyze.kernel
   def benchmark_matmul(n):
       """
       The simplest possible benchmark.
       We create two matrices and multiply them.
       """
       # Create two NxN matrices on GPU
       a = torch.randn(n, n, device="cuda")
       b = torch.randn(n, n, device="cuda")

       # Mark the kernel we want to profile
       with nsight.annotate("matmul"):
           c = a @ b

       return c

   if __name__ == "__main__":
       # Run the benchmark
       result = benchmark_matmul(1024)

That's it! Nsight Python will automatically profile your kernel, collect metrics, and display the results.

For more advanced examples including parameter sweeps, custom metrics, and visualization, check out the examples directory.
