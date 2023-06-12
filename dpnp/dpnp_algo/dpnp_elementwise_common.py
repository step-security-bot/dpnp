# cython: language_level=3
# distutils: language = c++
# -*- coding: utf-8 -*-
# *****************************************************************************
# Copyright (c) 2023, Intel Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# - Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
# - Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
# THE POSSIBILITY OF SUCH DAMAGE.
# *****************************************************************************


import dpnp
from dpnp.dpnp_array import dpnp_array
import dpnp.backend.extensions.vm._vm_impl as vmi

from dpctl.tensor._elementwise_common import (
    BinaryElementwiseFunc
)
import dpctl.tensor._tensor_impl as ti


__all__ = [
    "dpnp_divide"
]


_divide_docstring_ = """
divide(x1, x2, out=None, order='K')

Calculates the ratio for each element `x1_i` of the input array `x1` with
the respective element `x2_i` of the input array `x2`.

Args:
    x1 (dpnp.ndarray):
        First input array, expected to have numeric data type.
    x2 (dpnp.ndarray):
        Second input array, also expected to have numeric data type.
    out ({None, dpnp.ndarray}, optional):
        Output array to populate.
        Array have the correct shape and the expected data type.
    order ("C","F","A","K", None, optional):
        Memory layout of the newly output array, if parameter `out` is `None`.
        Default: "K".
Returns:
    dpnp.ndarray:
        an array containing the result of element-wise division. The data type
        of the returned array is determined by the Type Promotion Rules.
"""

def dpnp_divide(x1, x2, out=None, order='K'):
    """
    Invokes div() function from pybind11 extension of OneMKL VM if possible.
    Otherwise fully relies on dpctl.tensor implementation for divide() function.

    """

    def _call_divide(src1, src2, dst, sycl_queue, depends=[]):
        """A callback to register in BinaryElementwiseFunc class of dpctl.tensor"""

        if vmi._can_call_div(sycl_queue, src1, src2, dst):
            # call pybind11 extension for div() function from OneMKL VM
            return vmi._div(sycl_queue, src1, src2, dst, depends)
        return ti._divide(src1, src2, dst, sycl_queue, depends)

    # dpctl.tensor only works with usm_ndarray or scalar
    x1_usm_or_scalar = dpnp.get_usm_ndarray_or_scalar(x1)
    x2_usm_or_scalar = dpnp.get_usm_ndarray_or_scalar(x2)
    out_usm = None if out is None else dpnp.get_usm_ndarray(out)

    func = BinaryElementwiseFunc("divide", ti._divide_result_type, _call_divide, _divide_docstring_)
    res_usm = func(x1_usm_or_scalar, x2_usm_or_scalar, out=out_usm, order=order)
    return dpnp_array._create_from_usm_ndarray(res_usm)
