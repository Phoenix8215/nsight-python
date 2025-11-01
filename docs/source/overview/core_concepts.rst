.. SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: Apache-2.0

Core Concepts
=============

Nsight Python operates through three key primitives:

**1. Annotations**
An :func:`annotation <nsight.annotate>` wraps a region of code that launches GPU kernels and tags them for attribution.
Annotations can be used as decorators or context managers:

.. code-block:: python

   @nsight.annotate("torch")
   def torch_kernel():
       ...

   # or
   with nsight.annotate("cutlass4"):
       cutlass_kernel()

By default, each annotation is expected to contain a single kernel launch. For more detailed information about handling multiple kernels within an annotation, see the API documentation.

**2. Kernel Analysis Decorator**  
Use :func:`nsight.analyze.kernel` to annotate a benchmark function. Nsight Python will rerun this function one configuration at a time. You can provide configurations in two ways:

- **At decoration time** using the `configs` parameter.
- **At function call time** by passing `configs` directly as an argument when invoking the decorated function.

.. code-block:: python

   @nsight.analyze.kernel
   def benchmark(s):
       ...

   benchmark(configs=[(1024,), (2048,)])

**3. Plot Decorator**  
Add :func:`nsight.analyze.plot` to automatically generate plots from your profiling runs.

.. code-block:: python

   @nsight.analyze.plot(filename="plot.png", ylabel="Runtime (ns)")
   @nsight.analyze.kernel(configs=[(1024,), (2048,)])
   def benchmark(s):
       ...
