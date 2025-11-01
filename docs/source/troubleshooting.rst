.. SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: Apache-2.0

Troubleshooting
------------------

**Q: NVIDIA Nsight Compute CLI (ncu) is not found in my system path.**

Ensure that NVIDIA Nsight Compute is installed and its CLI (`ncu`) is added to your system's PATH. 
You can verify this by running:

.. code-block:: bash

   ncu --version

If the command is not recognized, add the NVIDIA Nsight Compute installation directory to your PATH.

**Q: Profiling fails with "Insufficient permissions" error.**

.. code-block:: bash

   ==ERROR== ERR_NVGPUCTRPERM - The user does not have permission to access NVIDIA GPU Performance Counters on the target device 0

This typically occurs when NVIDIA Nsight Compute does not have the required permissions to access the GPU. 
Make sure you have the necessary privileges to run profiling tools on your system. 
Follow NVIDIA Nsight Compute's documentation for instructions on how to set up permissions: `ERR_NVGPUCTRPERM Permission Issue <https://developer.nvidia.com/nvidia-development-tools-solutions-err_nvgpuctrperm-permission-issue-performance-counters>`_.


**Q: GPU throttling still occurs despite enabling failsafe.**

Verify that the `thermal_control` in the `nsight.analyze.kernel` decorator are set appropriately.
If the GPU temperature exceeds the configured thresholds, Nsight Python will attempt to wait for it to cool down.
Ensure your system's cooling setup is adequate for sustained GPU workloads.
