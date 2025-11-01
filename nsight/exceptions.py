# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

"""
Exceptions specific to Nsight Python profiling and analysis.
"""


class ProfilerException(Exception):
    """
    Exception raised for errors specific to the Profiler.

    Attributes:
        message: Explanation of the error.
    """

    pass


class NCUNotAvailableError(Exception):
    """
    Exception raised when NVIDIA Nsight Compute CLI (NCU) is not available or accessible.

    This can occur when:
    - NCU is not installed on the system
    - NCU is not in the system PATH
    - Required permissions are missing
    """

    pass


CUDA_CORE_UNAVAILABLE_MSG = "cuda-core is required for ignore_failures functionality.\n Install it with:\n  - pip install nsight-python[cu12]  (if you have CUDA 12.x)\n  - pip install nsight-python[cu13]  (if you have CUDA 13.x)"


@dataclass
class NCUErrorContext:
    """
    Context information for NCU error handling.

    Attributes:
        errors: The error logs from NCU
        log_file_path: Path to the NCU log file
        metric: The metric that was being collected
    """

    errors: list[str]
    log_file_path: str
    metric: str
