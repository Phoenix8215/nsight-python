.. SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: Apache-2.0

FAQ
------

**Q: Does Nsight Python support L2 purging?**

Yes, if you want it to. Nsight Python uses NCU to collect profiled data. NCU comes with
`Cache Control options <https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#cache-control>`_
that allow you to control L2 purging. By default Nsight Python follows NCU's default and
flushes all GPU caches before each run. This can be disabled by setting ``cache_control=None``
in the ``nsight.analyze.kernel`` decorator.


**Q: Does Nsight Python include kernel launch overheads?**

No, Nsight Python uses NCU-provided metrics that do not include kernel launch overheads.
If this is a desired feature, we could extend Nsight Python to support CUDA-event based
profiling as an alternative. Please file a feature request if you would like to see
this feature added.


**Q: Temparature-aware throttling prevention?! What does this mean and is it configurable?**

By default, Nsight Python continuously monitors the GPU temperatures during profiling runs
and ensures they stay within configurable threshold to avoid thermal throttling. 
This feature can be disabled by setting ``thermal_control=False`` in the ``nsight.analyze.kernel`` decorator.
