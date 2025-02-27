# cython: language_level=3
# cython: linetrace=True
# -*- coding: utf-8 -*-
# *****************************************************************************
# Copyright (c) 2016-2024, Intel Corporation
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

"""Module Backend

This module contains interface functions between C backend layer
and the rest of the library

"""

import numpy

from dpnp.dpnp_algo cimport *

import dpnp
import dpnp.dpnp_utils as utils_py

cimport numpy

cimport dpnp.dpnp_utils as utils

__all__ = [
    "dpnp_cond",
    "dpnp_eig",
    "dpnp_eigvals",
    "dpnp_matrix_rank",
    "dpnp_norm",
]


# C function pointer to the C library template functions
ctypedef c_dpctl.DPCTLSyclEventRef(*custom_linalg_1in_1out_func_ptr_t)(c_dpctl.DPCTLSyclQueueRef,
                                                                       void *, void * ,shape_elem_type * ,
                                                                       size_t, const c_dpctl.DPCTLEventVectorRef)
ctypedef c_dpctl.DPCTLSyclEventRef(*custom_linalg_1in_1out_func_ptr_t_)(c_dpctl.DPCTLSyclQueueRef,
                                                                        void * , void * , size_t * ,
                                                                        const c_dpctl.DPCTLEventVectorRef)
ctypedef c_dpctl.DPCTLSyclEventRef(*custom_linalg_1in_1out_with_size_func_ptr_t_)(c_dpctl.DPCTLSyclQueueRef,
                                                                                  void *, void * , size_t,
                                                                                  const c_dpctl.DPCTLEventVectorRef)
ctypedef c_dpctl.DPCTLSyclEventRef(*custom_linalg_1in_3out_shape_t)(c_dpctl.DPCTLSyclQueueRef,
                                                                    void *, void * , void * , void * ,
                                                                    size_t , size_t, const c_dpctl.DPCTLEventVectorRef)
ctypedef c_dpctl.DPCTLSyclEventRef(*custom_linalg_2in_1out_func_ptr_t)(c_dpctl.DPCTLSyclQueueRef,
                                                                       void *, void * , void * , size_t,
                                                                       const c_dpctl.DPCTLEventVectorRef)


cpdef object dpnp_cond(object input, object p):
    if p in ('f', 'fro'):
        # TODO: change order='K' when support is implemented
        input = dpnp.ravel(input, order='C')
        sqnorm = dpnp.dot(input, input)
        res = dpnp.sqrt(sqnorm)
        ret = dpnp.array([res])
    elif p == dpnp.inf:
        dpnp_sum_val = dpnp.sum(dpnp.abs(input), axis=1)
        ret = dpnp.max(dpnp_sum_val)
    elif p == -dpnp.inf:
        dpnp_sum_val = dpnp.sum(dpnp.abs(input), axis=1)
        ret = dpnp.min(dpnp_sum_val)
    elif p == 1:
        dpnp_sum_val = dpnp.sum(dpnp.abs(input), axis=0)
        ret = dpnp.max(dpnp_sum_val)
    elif p == -1:
        dpnp_sum_val = dpnp.sum(dpnp.abs(input), axis=0)
        ret = dpnp.min(dpnp_sum_val)
    else:
        ret = dpnp.array([input.item(0)])
    return ret


cpdef tuple dpnp_eig(utils.dpnp_descriptor x1):
    cdef shape_type_c x1_shape = x1.shape

    cdef size_t size = 0 if x1_shape.empty() else x1_shape.front()

    cdef DPNPFuncType param1_type = dpnp_dtype_to_DPNPFuncType(x1.dtype)
    cdef DPNPFuncData kernel_data = get_dpnp_function_ptr(DPNP_FN_EIG_EXT, param1_type, param1_type)

    x1_obj = x1.get_array()

    cdef (DPNPFuncType, void *) ret_type_and_func = utils.get_ret_type_and_func(kernel_data,
                                                                                x1_obj.sycl_device.has_aspect_fp64)
    cdef DPNPFuncType return_type = ret_type_and_func[0]
    cdef custom_linalg_2in_1out_func_ptr_t func = < custom_linalg_2in_1out_func_ptr_t > ret_type_and_func[1]

    cdef utils.dpnp_descriptor res_val = utils.create_output_descriptor((size,),
                                                                        return_type,
                                                                        None,
                                                                        device=x1_obj.sycl_device,
                                                                        usm_type=x1_obj.usm_type,
                                                                        sycl_queue=x1_obj.sycl_queue)
    cdef utils.dpnp_descriptor res_vec = utils.create_output_descriptor(x1_shape,
                                                                        return_type,
                                                                        None,
                                                                        device=x1_obj.sycl_device,
                                                                        usm_type=x1_obj.usm_type,
                                                                        sycl_queue=x1_obj.sycl_queue)

    result_sycl_queue = res_val.get_array().sycl_queue

    cdef c_dpctl.SyclQueue q = <c_dpctl.SyclQueue> result_sycl_queue
    cdef c_dpctl.DPCTLSyclQueueRef q_ref = q.get_queue_ref()

    # call FPTR function
    cdef c_dpctl.DPCTLSyclEventRef event_ref = func(q_ref,
                                                    x1.get_data(),
                                                    res_val.get_data(),
                                                    res_vec.get_data(),
                                                    size,
                                                    NULL)  # dep_events_ref

    with nogil: c_dpctl.DPCTLEvent_WaitAndThrow(event_ref)
    c_dpctl.DPCTLEvent_Delete(event_ref)

    return (res_val.get_pyobj(), res_vec.get_pyobj())


cpdef utils.dpnp_descriptor dpnp_eigvals(utils.dpnp_descriptor input):
    cdef shape_type_c input_shape = input.shape

    cdef size_t size = 0 if input_shape.empty() else input_shape.front()

    cdef DPNPFuncType param1_type = dpnp_dtype_to_DPNPFuncType(input.dtype)
    cdef DPNPFuncData kernel_data = get_dpnp_function_ptr(DPNP_FN_EIGVALS_EXT, param1_type, param1_type)

    input_obj = input.get_array()

    cdef (DPNPFuncType, void *) ret_type_and_func = utils.get_ret_type_and_func(kernel_data,
                                                                                input_obj.sycl_device.has_aspect_fp64)
    cdef DPNPFuncType return_type = ret_type_and_func[0]
    cdef custom_linalg_1in_1out_with_size_func_ptr_t_ func = < custom_linalg_1in_1out_with_size_func_ptr_t_ > ret_type_and_func[1]

    # create result array with type given by FPTR data
    cdef utils.dpnp_descriptor res_val = utils.create_output_descriptor((size,),
                                                                         return_type,
                                                                         None,
                                                                         device=input_obj.sycl_device,
                                                                         usm_type=input_obj.usm_type,
                                                                         sycl_queue=input_obj.sycl_queue)

    result_sycl_queue = res_val.get_array().sycl_queue

    cdef c_dpctl.SyclQueue q = <c_dpctl.SyclQueue> result_sycl_queue
    cdef c_dpctl.DPCTLSyclQueueRef q_ref = q.get_queue_ref()

    # call FPTR function
    cdef c_dpctl.DPCTLSyclEventRef event_ref = func(q_ref,
                                                    input.get_data(),
                                                    res_val.get_data(),
                                                    size,
                                                    NULL)  # dep_events_ref

    with nogil: c_dpctl.DPCTLEvent_WaitAndThrow(event_ref)
    c_dpctl.DPCTLEvent_Delete(event_ref)

    return res_val


cpdef utils.dpnp_descriptor dpnp_matrix_rank(utils.dpnp_descriptor input):
    cdef shape_type_c input_shape = input.shape
    cdef DPNPFuncType param1_type = dpnp_dtype_to_DPNPFuncType(input.dtype)

    cdef DPNPFuncData kernel_data = get_dpnp_function_ptr(DPNP_FN_MATRIX_RANK_EXT, param1_type, param1_type)

    input_obj = input.get_array()

    # create result array with type given by FPTR data
    cdef utils.dpnp_descriptor result = utils.create_output_descriptor((1,),
                                                                       kernel_data.return_type,
                                                                       None,
                                                                       device=input_obj.sycl_device,
                                                                       usm_type=input_obj.usm_type,
                                                                       sycl_queue=input_obj.sycl_queue)

    result_sycl_queue = result.get_array().sycl_queue

    cdef c_dpctl.SyclQueue q = <c_dpctl.SyclQueue> result_sycl_queue
    cdef c_dpctl.DPCTLSyclQueueRef q_ref = q.get_queue_ref()

    cdef custom_linalg_1in_1out_func_ptr_t func = <custom_linalg_1in_1out_func_ptr_t > kernel_data.ptr

    cdef c_dpctl.DPCTLSyclEventRef event_ref = func(q_ref,
                                                    input.get_data(),
                                                    result.get_data(),
                                                    input_shape.data(),
                                                    input.ndim,
                                                    NULL)  # dep_events_ref

    with nogil: c_dpctl.DPCTLEvent_WaitAndThrow(event_ref)
    c_dpctl.DPCTLEvent_Delete(event_ref)

    return result


cpdef object dpnp_norm(object input, ord=None, axis=None):
    cdef long size_input = input.size
    cdef shape_type_c shape_input = input.shape

    dev = input.get_array().sycl_device
    if input.dtype == dpnp.float32 or not dev.has_aspect_fp64:
        res_type = dpnp.float32
    else:
        res_type = dpnp.float64

    if size_input == 0:
        return dpnp.array([dpnp.nan], dtype=res_type)

    if isinstance(axis, int):
        axis_ = tuple([axis])
    else:
        axis_ = axis

    ndim = input.ndim
    if axis is None:
        if ((ord is None) or
            (ord in ('f', 'fro') and ndim == 2) or
                (ord == 2 and ndim == 1)):

            # TODO: change order='K' when support is implemented
            input = dpnp.ravel(input, order='C')
            sqnorm = dpnp.dot(input, input)
            ret = dpnp.sqrt([sqnorm], dtype=res_type)
            return dpnp.array(ret.reshape(1, *ret.shape), dtype=res_type)

    len_axis = 1 if axis is None else len(axis_)
    if len_axis == 1:
        if ord == dpnp.inf:
            return dpnp.array([dpnp.abs(input).max(axis=axis)])
        elif ord == -dpnp.inf:
            return dpnp.array([dpnp.abs(input).min(axis=axis)])
        elif ord == 0:
            return input.dtype.type(dpnp.count_nonzero(input, axis=axis))
        elif ord is None or ord == 2:
            s = input * input
            return dpnp.sqrt(dpnp.sum(s, axis=axis), dtype=res_type)
        elif isinstance(ord, str):
            raise ValueError(f"Invalid norm order '{ord}' for vectors")
        else:
            absx = dpnp.abs(input)
            absx_size = absx.size
            absx_power = utils_py.create_output_descriptor_py((absx_size,), absx.dtype, None).get_pyobj()

            absx_flatiter = absx.flat

            for i in range(absx_size):
                absx_elem = absx_flatiter[i]
                absx_power[i] = absx_elem ** ord
            absx_ = dpnp.reshape(absx_power, absx.shape)
            ret = dpnp.sum(absx_, axis=axis)
            ret_size = ret.size
            ret_power = utils_py.create_output_descriptor_py((ret_size,), None, None).get_pyobj()

            ret_flatiter = ret.flat

            for i in range(ret_size):
                ret_elem = ret_flatiter[i]
                ret_power[i] = ret_elem ** (1 / ord)
            ret_ = dpnp.reshape(ret_power, ret.shape)
            return ret_
    elif len_axis == 2:
        row_axis, col_axis = axis_
        if row_axis == col_axis:
            raise ValueError('Duplicate axes given.')
        # if ord == 2:
        #     ret =  _multi_svd_norm(input, row_axis, col_axis, amax)
        # elif ord == -2:
        #     ret = _multi_svd_norm(input, row_axis, col_axis, amin)
        elif ord == 1:
            if col_axis > row_axis:
                col_axis -= 1
            dpnp_sum_val = dpnp.sum(dpnp.abs(input), axis=row_axis)
            ret = dpnp_sum_val.min(axis=col_axis)
        elif ord == dpnp.inf:
            if row_axis > col_axis:
                row_axis -= 1
            dpnp_sum_val = dpnp.sum(dpnp.abs(input), axis=col_axis)
            ret = dpnp_sum_val.max(axis=row_axis)
        elif ord == -1:
            if col_axis > row_axis:
                col_axis -= 1
            dpnp_sum_val = dpnp.sum(dpnp.abs(input), axis=row_axis)
            ret = dpnp_sum_val.min(axis=col_axis)
        elif ord == -dpnp.inf:
            if row_axis > col_axis:
                row_axis -= 1
            dpnp_sum_val = dpnp.sum(dpnp.abs(input), axis=col_axis)
            ret = dpnp_sum_val.min(axis=row_axis)
        elif ord in [None, 'fro', 'f']:
            ret = dpnp.sqrt(dpnp.sum(input * input, axis=axis))
        # elif ord == 'nuc':
        #     ret = _multi_svd_norm(input, row_axis, col_axis, sum)
        else:
            raise ValueError("Invalid norm order for matrices.")

        return ret
    else:
        raise ValueError("Improper number of dimensions to norm.")
