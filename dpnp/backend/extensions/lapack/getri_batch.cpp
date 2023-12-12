//*****************************************************************************
// Copyright (c) 2023, Intel Corporation
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
// - Redistributions of source code must retain the above copyright notice,
//   this list of conditions and the following disclaimer.
// - Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the following disclaimer in the documentation
//   and/or other materials provided with the distribution.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
// THE POSSIBILITY OF SUCH DAMAGE.
//*****************************************************************************

#include <pybind11/pybind11.h>

// dpctl tensor headers
#include "utils/memory_overlap.hpp"
#include "utils/type_utils.hpp"

#include "getri.hpp"
#include "types_matrix.hpp"

#include "dpnp_utils.hpp"

namespace dpnp
{
namespace backend
{
namespace ext
{
namespace lapack
{
namespace mkl_lapack = oneapi::mkl::lapack;
namespace py = pybind11;
namespace type_utils = dpctl::tensor::type_utils;

typedef sycl::event (*getri_batch_impl_fn_ptr_t)(
    sycl::queue,
    std::int64_t,
    char *,
    std::int64_t,
    std::int64_t,
    std::int64_t *,
    std::int64_t,
    std::int64_t,
    std::int64_t *,
    std::vector<sycl::event> &,
    const std::vector<sycl::event> &);

static getri_batch_impl_fn_ptr_t
    getri_batch_dispatch_vector[dpctl_td_ns::num_types];

template <typename T>
static sycl::event getri_batch_impl(sycl::queue exec_q,
                                    std::int64_t n,
                                    char *in_a,
                                    std::int64_t lda,
                                    std::int64_t stride_a,
                                    std::int64_t *ipiv,
                                    std::int64_t stride_ipiv,
                                    std::int64_t batch_size,
                                    std::int64_t *dev_info,
                                    std::vector<sycl::event> &host_task_events,
                                    const std::vector<sycl::event> &depends)
{
    type_utils::validate_type_for_device<T>(exec_q);

    T *a = reinterpret_cast<T *>(in_a);

    const std::int64_t scratchpad_size =
        mkl_lapack::getri_batch_scratchpad_size<T>(exec_q, n, lda, stride_a,
                                                   stride_ipiv, batch_size);
    T *scratchpad = nullptr;

    std::stringstream error_msg;
    std::int64_t info = 0;

    sycl::event getri_batch_event;
    try {
        scratchpad = sycl::malloc_device<T>(scratchpad_size, exec_q);

        getri_batch_event = mkl_lapack::getri_batch(
            exec_q,
            n,           // Order of each square matrix in the batch; (0 ≤ n).
            a,           // Pointer to the batch of matrices.
            lda,         // The leading dimension of `a`.
            stride_a,    // Stride between matrices: Element spacing between
                         // matrices in `a`.
            ipiv,        // Pivot indices: Pointer to the pivot indices for each
                         // matrix.
            stride_ipiv, // Stride between pivot indices: Spacing between pivot
                         // arrays in 'ipiv'.
            batch_size,  // Total number of matrices in the batch.
            scratchpad,  // Pointer to scratchpad memory to be used by MKL
                         // routine for storing intermediate results.
            scratchpad_size, depends);
    } catch (mkl_lapack::exception const &e) {
        error_msg
            << "Unexpected MKL exception caught during getri() call:\nreason: "
            << e.what() << "\ninfo: " << e.info();
        info = e.info();
    } catch (sycl::exception const &e) {
        error_msg << "Unexpected SYCL exception caught during getri() call:\n"
                  << e.what();
        info = -1;
    }

    if (info != 0) // an unexpected error occurs
    {
        if (scratchpad != nullptr) {
            sycl::free(scratchpad, exec_q);
        }

        if (info < 0) {
            throw std::runtime_error(error_msg.str());
        }
    }

    sycl::event write_info_event = exec_q.submit([&](sycl::handler &cgh) {
        cgh.depends_on(getri_batch_event);
        cgh.single_task([=]() { dev_info[0] = info; });
    });

    host_task_events.push_back(write_info_event);

    sycl::event clean_up_event = exec_q.submit([&](sycl::handler &cgh) {
        cgh.depends_on(getri_batch_event);
        auto ctx = exec_q.get_context();
        cgh.host_task([ctx, scratchpad]() { sycl::free(scratchpad, ctx); });
    });
    host_task_events.push_back(clean_up_event);
    return getri_batch_event;
}

std::pair<sycl::event, sycl::event>
    getri_batch(sycl::queue q,
                dpctl::tensor::usm_ndarray a_array,
                dpctl::tensor::usm_ndarray ipiv_array,
                dpctl::tensor::usm_ndarray dev_info_array,
                std::int64_t n,
                std::int64_t stride_a,
                std::int64_t stride_ipiv,
                std::int64_t batch_size,
                const std::vector<sycl::event> &depends)
{

    auto array_types = dpctl_td_ns::usm_ndarray_types();
    int a_array_type_id =
        array_types.typenum_to_lookup_id(a_array.get_typenum());

    getri_batch_impl_fn_ptr_t getri_batch_fn =
        getri_batch_dispatch_vector[a_array_type_id];
    if (getri_batch_fn == nullptr) {
        throw py::value_error(
            "No getri_batch implementation defined for the provided type "
            "of the input matrix.");
    }

    char *a_array_data = a_array.get_data();
    const std::int64_t lda = std::max<size_t>(1UL, n);

    char *ipiv_array_data = ipiv_array.get_data();
    std::int64_t *d_ipiv = reinterpret_cast<std::int64_t *>(ipiv_array_data);

    char *dev_info_array_data = dev_info_array.get_data();
    std::int64_t *d_dev_info =
        reinterpret_cast<std::int64_t *>(dev_info_array_data);

    std::vector<sycl::event> host_task_events;
    sycl::event getri_batch_ev =
        getri_batch_fn(q, n, a_array_data, lda, stride_a, d_ipiv, stride_ipiv,
                       batch_size, d_dev_info, host_task_events, depends);

    sycl::event args_ev = dpctl::utils::keep_args_alive(
        q, {a_array, ipiv_array, dev_info_array}, host_task_events);

    return std::make_pair(args_ev, getri_batch_ev);
}

template <typename fnT, typename T>
struct GetriBatchContigFactory
{
    fnT get()
    {
        if constexpr (types::GetriBatchTypePairSupportFactory<T>::is_defined) {
            return getri_batch_impl<T>;
        }
        else {
            return nullptr;
        }
    }
};

void init_getri_batch_dispatch_vector(void)
{
    dpctl_td_ns::DispatchVectorBuilder<getri_batch_impl_fn_ptr_t,
                                       GetriBatchContigFactory,
                                       dpctl_td_ns::num_types>
        contig;
    contig.populate_dispatch_vector(getri_batch_dispatch_vector);
}
} // namespace lapack
} // namespace ext
} // namespace backend
} // namespace dpnp
