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

import dpctl.tensor as dpt

import dpnp


def _get_unwrapped_index_key(key):
    """
    Get an unwrapped index key.

    Return a key where each nested instance of DPNP array is unwrapped into USM ndarray
    for further processing in DPCTL advanced indexing functions.

    """

    if isinstance(key, tuple):
        if any(isinstance(x, dpnp_array) for x in key):
            # create a new tuple from the input key with unwrapped DPNP arrays
            return tuple(
                x.get_array() if isinstance(x, dpnp_array) else x for x in key
            )
    elif isinstance(key, dpnp_array):
        return key.get_array()
    return key


class dpnp_array:
    """
    Multi-dimensional array object.

    This is a wrapper around dpctl.tensor.usm_ndarray that provides
    methods to be compliant with original Numpy.

    """

    def __init__(
        self,
        shape,
        dtype=None,
        buffer=None,
        offset=0,
        strides=None,
        order="C",
        device=None,
        usm_type="device",
        sycl_queue=None,
    ):
        if buffer is not None:
            if not isinstance(buffer, dpt.usm_ndarray):
                raise TypeError(
                    "Expected dpctl.tensor.usm_ndarray, got {}"
                    "".format(type(buffer))
                )
            if buffer.shape != shape:
                raise ValueError(
                    "Expected buffer.shape={}, got {}"
                    "".format(shape, buffer.shape)
                )
            self._array_obj = dpt.asarray(buffer, copy=False, order=order)
        else:
            sycl_queue_normalized = dpnp.get_normalized_queue_device(
                device=device, sycl_queue=sycl_queue
            )
            self._array_obj = dpt.usm_ndarray(
                shape,
                dtype=dtype,
                strides=strides,
                buffer=usm_type,
                offset=offset,
                order=order,
                buffer_ctor_kwargs={"queue": sycl_queue_normalized},
            )

    @property
    def __sycl_usm_array_interface__(self):
        return self._array_obj.__sycl_usm_array_interface__

    def get_array(self):
        """Get usm_ndarray object."""
        return self._array_obj

    @property
    def T(self):
        """View of the transposed array."""
        return self.transpose()

    def to_device(self, target_device):
        """Transfer array to target device."""

        return dpnp_array(
            shape=self.shape, buffer=self.get_array().to_device(target_device)
        )

    @property
    def sycl_queue(self):
        return self._array_obj.sycl_queue

    @property
    def sycl_device(self):
        return self._array_obj.sycl_device

    @property
    def sycl_context(self):
        return self._array_obj.sycl_context

    @property
    def device(self):
        return self._array_obj.device

    @property
    def usm_type(self):
        return self._array_obj.usm_type

    def __abs__(self):
        r"""Return \|self\|."""
        return dpnp.abs(self)

    def __add__(self, other):
        """Return self+value."""
        return dpnp.add(self, other)

    def __and__(self, other):
        """Return self&value."""
        return dpnp.bitwise_and(self, other)

    # '__array__',
    # '__array_finalize__',
    # '__array_function__',
    # '__array_interface__',
    # '__array_prepare__',
    # '__array_priority__',
    # '__array_struct__',
    # '__array_ufunc__',
    # '__array_wrap__',

    def __bool__(self):
        return self._array_obj.__bool__()

    # '__class__',

    def __complex__(self):
        return self._array_obj.__complex__()

    # '__contains__',

    def __copy__(self):
        """
        Used if :func:`copy.copy` is called on an array. Returns a copy of the array.

        Equivalent to ``a.copy(order="K")``.
        """
        return self.copy(order="K")

    # '__deepcopy__',
    # '__delattr__',
    # '__delitem__',
    # '__dir__',
    # '__divmod__',
    # '__doc__',

    def __dlpack__(self, stream=None):
        return self._array_obj.__dlpack__(stream=stream)

    def __dlpack_device__(self):
        return self._array_obj.__dlpack_device__()

    def __eq__(self, other):
        return dpnp.equal(self, other)

    def __float__(self):
        return self._array_obj.__float__()

    def __floordiv__(self, other):
        """Return self//value."""
        return dpnp.floor_divide(self, other)

    # '__format__',

    def __ge__(self, other):
        return dpnp.greater_equal(self, other)

    # '__getattribute__',

    def __getitem__(self, key):
        """Return self[key]."""
        key = _get_unwrapped_index_key(key)

        item = self._array_obj.__getitem__(key)
        if not isinstance(item, dpt.usm_ndarray):
            raise RuntimeError(
                "Expected dpctl.tensor.usm_ndarray, got {}"
                "".format(type(item))
            )

        res = self.__new__(dpnp_array)
        res._array_obj = item

        return res

    def __gt__(self, other):
        return dpnp.greater(self, other)

    # '__hash__',

    def __iadd__(self, other):
        """Return self+=value."""
        dpnp.add(self, other, out=self)
        return self

    def __iand__(self, other):
        """Return self&=value."""
        dpnp.bitwise_and(self, other, out=self)
        return self

    def __ifloordiv__(self, other):
        """Return self//=value."""
        dpnp.floor_divide(self, other, out=self)
        return self

    def __ilshift__(self, other):
        """Return self<<=value."""
        dpnp.left_shift(self, other, out=self)
        return self

    # '__imatmul__',

    def __imod__(self, other):
        """Return self%=value."""
        dpnp.remainder(self, other, out=self)
        return self

    def __imul__(self, other):
        """Return self*=value."""
        dpnp.multiply(self, other, out=self)
        return self

    def __index__(self):
        return self._array_obj.__index__()

    # '__init__',
    # '__init_subclass__',

    def __int__(self):
        return self._array_obj.__int__()

    def __invert__(self):
        """Return ~self."""
        return dpnp.invert(self)

    def __ior__(self, other):
        """Return self|=value."""
        dpnp.bitwise_or(self, other, out=self)
        return self

    def __ipow__(self, other):
        """Return self**=value."""
        dpnp.power(self, other, out=self)
        return self

    def __irshift__(self, other):
        """Return self>>=value."""
        dpnp.right_shift(self, other, out=self)
        return self

    def __isub__(self, other):
        """Return self-=value."""
        dpnp.subtract(self, other, out=self)
        return self

    # '__iter__',

    def __itruediv__(self, other):
        """Return self/=value."""
        dpnp.true_divide(self, other, out=self)
        return self

    def __ixor__(self, other):
        """Return self^=value."""
        dpnp.bitwise_xor(self, other, out=self)
        return self

    def __le__(self, other):
        return dpnp.less_equal(self, other)

    def __len__(self):
        """Return len(self)."""

        return self._array_obj.__len__()

    def __lshift__(self, other):
        """Return self<<value."""
        return dpnp.left_shift(self, other)

    def __lt__(self, other):
        return dpnp.less(self, other)

    def __matmul__(self, other):
        return dpnp.matmul(self, other)

    def __mod__(self, other):
        """Return self%value."""
        return dpnp.remainder(self, other)

    def __mul__(self, other):
        """Return self*value."""
        return dpnp.multiply(self, other)

    def __ne__(self, other):
        return dpnp.not_equal(self, other)

    def __neg__(self):
        """Return -self."""
        return dpnp.negative(self)

    # '__new__',

    def __or__(self, other):
        """Return self|value."""
        return dpnp.bitwise_or(self, other)

    def __pos__(self):
        """Return +self."""
        return dpnp.positive(self)

    def __pow__(self, other):
        """Return self**value."""
        return dpnp.power(self, other)

    def __radd__(self, other):
        return dpnp.add(other, self)

    def __rand__(self, other):
        return dpnp.bitwise_and(other, self)

    # '__rdivmod__',
    # '__reduce__',
    # '__reduce_ex__',

    def __repr__(self):
        return dpt.usm_ndarray_repr(self._array_obj, prefix="array")

    def __rfloordiv__(self, other):
        return dpnp.floor_divide(self, other)

    def __rlshift__(self, other):
        return dpnp.left_shift(other, self)

    def __rmatmul__(self, other):
        return dpnp.matmul(other, self)

    def __rmod__(self, other):
        return dpnp.remainder(other, self)

    def __rmul__(self, other):
        return dpnp.multiply(other, self)

    def __ror__(self, other):
        return dpnp.bitwise_or(other, self)

    def __rpow__(self, other):
        return dpnp.power(other, self)

    def __rrshift__(self, other):
        return dpnp.right_shift(other, self)

    def __rshift__(self, other):
        """Return self>>value."""
        return dpnp.right_shift(self, other)

    def __rsub__(self, other):
        return dpnp.subtract(other, self)

    def __rtruediv__(self, other):
        return dpnp.true_divide(other, self)

    def __rxor__(self, other):
        return dpnp.bitwise_xor(other, self)

    # '__setattr__',

    def __setitem__(self, key, val):
        """Set self[key] to value."""
        key = _get_unwrapped_index_key(key)

        if isinstance(val, dpnp_array):
            val = val.get_array()

        self._array_obj.__setitem__(key, val)

    # '__setstate__',
    # '__sizeof__',

    def __str__(self):
        """
        Output values from the array to standard output.

        Examples
        --------
        >>> print(a)
        [[ 136.  136.  136.]
         [ 272.  272.  272.]
         [ 408.  408.  408.]]

        """

        return self._array_obj.__str__()

    def __sub__(self, other):
        """Return self-value."""
        return dpnp.subtract(self, other)

    # '__subclasshook__',

    def __truediv__(self, other):
        """Return self/value."""
        return dpnp.true_divide(self, other)

    def __xor__(self, other):
        """Return self^value."""
        return dpnp.bitwise_xor(self, other)

    @staticmethod
    def _create_from_usm_ndarray(usm_ary: dpt.usm_ndarray):
        if not isinstance(usm_ary, dpt.usm_ndarray):
            raise TypeError(
                f"Expected dpctl.tensor.usm_ndarray, got {type(usm_ary)}"
            )
        res = dpnp_array.__new__(dpnp_array)
        res._array_obj = usm_ary
        return res

    def all(self, axis=None, out=None, keepdims=False, *, where=True):
        """
        Returns True if all elements evaluate to True.

        Refer to :obj:`dpnp.all` for full documentation.

        See Also
        --------
        :obj:`dpnp.all` : equivalent function

        """

        return dpnp.all(
            self, axis=axis, out=out, keepdims=keepdims, where=where
        )

    def any(self, axis=None, out=None, keepdims=False, *, where=True):
        """
        Returns True if any of the elements of `a` evaluate to True.

        Refer to :obj:`dpnp.any` for full documentation.

        See Also
        --------
        :obj:`dpnp.any` : equivalent function

        """

        return dpnp.any(
            self, axis=axis, out=out, keepdims=keepdims, where=where
        )

    def argmax(self, axis=None, out=None, *, keepdims=False):
        """
        Returns array of indices of the maximum values along the given axis.

        Refer to :obj:`dpnp.argmax` for full documentation.

        """
        return dpnp.argmax(self, axis, out, keepdims=keepdims)

    def argmin(self, axis=None, out=None, *, keepdims=False):
        """
        Return array of indices to the minimum values along the given axis.

        Refer to :obj:`dpnp.argmin` for full documentation.

        """
        return dpnp.argmin(self, axis, out, keepdims=keepdims)

    # 'argpartition',

    def argsort(self, axis=-1, kind=None, order=None):
        """
        Return an ndarray of indices that sort the array along the specified axis.

        Refer to :obj:`dpnp.argsort` for full documentation.

        """
        return dpnp.argsort(self, axis, kind, order)

    def asnumpy(self):
        """
        Copy content of the array into :class:`numpy.ndarray` instance of the same shape and data type.

        Returns
        -------
        numpy.ndarray
            An instance of :class:`numpy.ndarray` populated with the array content.

        """

        return dpt.asnumpy(self._array_obj)

    def astype(self, dtype, order="K", casting="unsafe", subok=True, copy=True):
        """
        Copy the array with data type casting.

        For full documentation refer to :obj:`numpy.ndarray.astype`.

        Parameters
        ----------
        x1 : {dpnp.ndarray, usm_ndarray}
            Array data type casting.
        dtype : dtype
            Target data type.
        order : {'C', 'F', 'A', 'K'}
            Row-major (C-style) or column-major (Fortran-style) order.
            When ``order`` is 'A', it uses 'F' if ``a`` is column-major and uses 'C' otherwise.
            And when ``order`` is 'K', it keeps strides as closely as possible.
        copy : bool
            If it is False and no cast happens, then this method returns the array itself.
            Otherwise, a copy is returned.
        casting : {'no', 'equiv', 'safe', 'same_kind', 'unsafe'}, optional
            Controls what kind of data casting may occur. Defaults to 'unsafe' for backwards compatibility.
            'no' means the data types should not be cast at all.
            'equiv' means only byte-order changes are allowed.
            'safe' means only casts which can preserve values are allowed.
            'same_kind' means only safe casts or casts within a kind, like float64 to float32, are allowed.
            'unsafe' means any data conversions may be done.
        copy : bool, optional
            By default, astype always returns a newly allocated array. If this is set to false, and the dtype,
            order, and subok requirements are satisfied, the input array is returned instead of a copy.

        Returns
        -------
        arr_t : dpnp.ndarray
            Unless `copy` is ``False`` and the other conditions for returning the input array
            are satisfied, `arr_t` is a new array of the same shape as the input array,
            with dtype, order given by dtype, order.

        Limitations
        -----------
        Parameter `subok` is supported with default value.
        Otherwise ``NotImplementedError`` exception will be raised.

        Examples
        --------
        >>> import dpnp as np
        >>> x = np.array([1, 2, 2.5])
        >>> x
        array([1. , 2. , 2.5])
        >>> x.astype(int)
        array([1, 2, 2])

        """

        if subok is not True:
            raise NotImplementedError(
                f"subok={subok} is currently not supported"
            )

        return dpnp.astype(self, dtype, order=order, casting=casting, copy=copy)

    # 'base',
    # 'byteswap',

    def choose(input, choices, out=None, mode="raise"):
        """Construct an array from an index array and a set of arrays to choose from."""

        return dpnp.choose(input, choices, out, mode)

    def clip(self, min=None, max=None, out=None, **kwargs):
        """
        Clip (limit) the values in an array.

        Refer to :obj:`dpnp.clip` for full documentation.

        """

        return dpnp.clip(self, min, max, out=out, **kwargs)

    # 'compress',

    def conj(self):
        """
        Complex-conjugate all elements.

        For full documentation refer to :obj:`numpy.ndarray.conj`.

        """

        if not dpnp.issubsctype(self.dtype, dpnp.complexfloating):
            return self
        else:
            return dpnp.conjugate(self)

    def conjugate(self):
        """
        Return the complex conjugate, element-wise.

        For full documentation refer to :obj:`numpy.ndarray.conjugate`.

        """

        if not dpnp.issubsctype(self.dtype, dpnp.complexfloating):
            return self
        else:
            return dpnp.conjugate(self)

    def copy(self, order="C"):
        """
        Return a copy of the array.

        Returns
        -------
        out : dpnp.ndarray
            A copy of the array.

        See also
        --------
        :obj:`dpnp.copy` : Similar function with different default behavior
        :obj:`dpnp.copyto` : Copies values from one array to another.

        Notes
        -----
        This function is the preferred method for creating an array copy. The
        function :func:`dpnp.copy` is similar, but it defaults to using order 'K'.

        Examples
        --------
        >>> import dpnp as np
        >>> x = np.array([[1, 2, 3], [4, 5, 6]], order='F')
        >>> y = x.copy()
        >>> x.fill(0)

        >>> x
        array([[0, 0, 0],
               [0, 0, 0]])

        >>> y
        array([[1, 2, 3],
               [4, 5, 6]])

        >>> y.flags['C_CONTIGUOUS']
        True

        """

        return dpnp.copy(self, order=order)

    # 'ctypes',
    # 'cumprod',

    def cumsum(self, axis=None, dtype=None, out=None):
        """
        Return the cumulative sum of the elements along the given axis.

        See Also
        --------
        :obj:`dpnp.cumsum`

        """

        return dpnp.cumsum(self, axis=axis, dtype=dtype, out=out)

    # 'data',

    def diagonal(input, offset=0, axis1=0, axis2=1):
        """
        Return specified diagonals.

        See Also
        --------
        :obj:`dpnp.diagonal`

        """

        return dpnp.diagonal(input, offset, axis1, axis2)

    def dot(self, b, out=None):
        """
        Dot product of two arrays.

        For full documentation refer to :obj:`dpnp.dot`.

        Examples
        --------
        >>> import dpnp as np
        >>> a = np.eye(2)
        >>> b = np.ones((2, 2)) * 2
        >>> a.dot(b)
        array([[2., 2.],
               [2., 2.]])

        This array method can be conveniently chained:

        >>> a.dot(b).dot(b)
        array([[8., 8.],
               [8., 8.]])
        """

        return dpnp.dot(self, b, out)

    @property
    def dtype(self):
        """Returns NumPy's dtype corresponding to the type of the array elements."""

        return self._array_obj.dtype

    # 'dump',
    # 'dumps',

    def fill(self, value):
        """
        Fill the array with a scalar value.

        Parameters
        ----------
        value : scalar
            All elements of `a` will be assigned this value.

        Examples
        --------
        >>> a = np.array([1, 2])
        >>> a.fill(0)
        >>> a
        array([0, 0])
        >>> a = np.empty(2)
        >>> a.fill(1)
        >>> a
        array([1.,  1.])

        """

        for i in range(self.size):
            self.flat[i] = value

    @property
    def flags(self):
        """Return information about the memory layout of the array."""

        return self._array_obj.flags

    @property
    def flat(self):
        """Return a flat iterator, or set a flattened version of self to value."""

        return dpnp.flatiter(self)

    def flatten(self, order="C"):
        """
        Return a copy of the array collapsed into one dimension.

        Parameters
        ----------
        order: {'C', 'F', 'A', 'K'}, optional
            'C' means to flatten in row-major (C-style) order.
            'F' means to flatten in column-major (Fortran- style) order.
            'A' means to flatten in column-major order if a is Fortran contiguous in memory, row-major order otherwise.
            'K' means to flatten a in the order the elements occur in memory. The default is 'C'.

        Returns
        -------
        out: ndarray
            A copy of the input array, flattened to one dimension.

        See Also
        --------
        :obj:`dpnp.ravel`, :obj:`dpnp.flat`

        """
        new_arr = self.__new__(dpnp_array)
        new_arr._array_obj = dpt.empty(
            self.shape,
            dtype=self.dtype,
            order=order,
            device=self._array_obj.sycl_device,
            usm_type=self._array_obj.usm_type,
            sycl_queue=self._array_obj.sycl_queue,
        )

        if self.size > 0:
            dpt._copy_utils._copy_from_usm_ndarray_to_usm_ndarray(
                new_arr._array_obj, self._array_obj
            )
            new_arr._array_obj = dpt.reshape(new_arr._array_obj, (self.size,))

        return new_arr

    # 'getfield',

    @property
    def imag(self):
        """
        The imaginary part of the array.

        For full documentation refer to :obj:`numpy.ndarray.imag`.

        Examples
        --------
        >>> import dpnp as np
        >>> x = np.sqrt(np.array([1+0j, 0+1j]))
        >>> x.imag
        array([0.        , 0.70710677])

        """
        return dpnp.imag(self)

    @imag.setter
    def imag(self, value):
        """
        Set the imaginary part of the array.

        For full documentation refer to :obj:`numpy.ndarray.imag`.

        Examples
        --------
        >>> import dpnp as np
        >>> a = np.array([1+2j, 3+4j, 5+6j])
        >>> a.imag = 9
        >>> a
        array([1.+9.j, 3.+9.j, 5.+9.j])

        """
        if dpnp.issubsctype(self.dtype, dpnp.complexfloating):
            dpnp.copyto(self._array_obj.imag, value)
        else:
            raise TypeError("array does not have imaginary part to set")

    def item(self, id=None):
        """
        Copy an element of an array to a standard Python scalar and return it.

        For full documentation refer to :obj:`numpy.ndarray.item`.

        Examples
        --------
        >>> np.random.seed(123)
        >>> x = np.random.randint(9, size=(3, 3))
        >>> x
        array([[2, 2, 6],
               [1, 3, 6],
               [1, 0, 1]])
        >>> x.item(3)
        1
        >>> x.item(7)
        0
        >>> x.item((0, 1))
        2
        >>> x.item((2, 2))
        1

        """

        if id is None:
            if self.size != 1:
                raise ValueError(
                    "DPNP dparray::item(): can only convert an array of size 1 to a Python scalar"
                )
            else:
                id = 0

        return self.flat[id]

    # 'itemset',

    @property
    def itemsize(self):
        """Size of one array element in bytes."""

        return self._array_obj.itemsize

    def max(
        self,
        axis=None,
        out=None,
        keepdims=False,
        initial=None,
        where=True,
    ):
        """
        Return the maximum along an axis.

        Refer to :obj:`dpnp.max` for full documentation.
        """

        return dpnp.max(self, axis, out, keepdims, initial, where)

    def mean(
        self, axis=None, dtype=None, out=None, keepdims=False, *, where=True
    ):
        """
        Returns the average of the array elements.

        Refer to :obj:`dpnp.mean` for full documentation.
        """

        return dpnp.mean(self, axis, dtype, out, keepdims, where=where)

    def min(
        self,
        axis=None,
        out=None,
        keepdims=False,
        initial=None,
        where=True,
    ):
        """
        Return the minimum along a given axis.

        Refer to :obj:`dpnp.min` for full documentation.
        """

        return dpnp.min(self, axis, out, keepdims, initial, where)

    @property
    def nbytes(self):
        """Total bytes consumed by the elements of the array."""

        return self._array_obj.nbytes

    @property
    def ndim(self):
        """Number of array dimensions."""

        return self._array_obj.ndim

    # 'newbyteorder',

    def nonzero(self):
        """Return the indices of the elements that are non-zero."""

        return dpnp.nonzero(self)

    def partition(self, kth, axis=-1, kind="introselect", order=None):
        """
        Return a partitioned copy of an array.

        Rearranges the elements in the array in such a way that the value of the
        element in kth position is in the position it would be in a sorted array.

        All elements smaller than the kth element are moved before this element and
        all equal or greater are moved behind it. The ordering of the elements in
        the two partitions is undefined.

        Refer to `dpnp.partition` for full documentation.

        See Also
        --------
        :obj:`dpnp.partition` : Return a partitioned copy of an array.

        Examples
        --------
        >>> import dpnp as np
        >>> a = np.array([3, 4, 2, 1])
        >>> a.partition(3)
        >>> a
        array([1, 2, 3, 4])

        """

        self._array_obj = dpnp.partition(
            self, kth, axis=axis, kind=kind, order=order
        ).get_array()

    def prod(
        self,
        axis=None,
        dtype=None,
        out=None,
        keepdims=False,
        initial=None,
        where=True,
    ):
        """
        Returns the prod along a given axis.

        For full documentation refer to :obj:`dpnp.prod`.

        """

        return dpnp.prod(self, axis, dtype, out, keepdims, initial, where)

    def put(self, indices, vals, /, *, axis=None, mode="wrap"):
        """
        Puts values of an array into another array along a given axis.

        For full documentation refer to :obj:`numpy.put`.
        """

        return dpnp.put(self, indices, vals, axis=axis, mode=mode)

    def ravel(self, order="C"):
        """
        Return a contiguous flattened array.

        For full documentation refer to :obj:`dpnp.ravel`.

        """

        return dpnp.ravel(self, order=order)

    @property
    def real(self):
        """
        The real part of the array.

        For full documentation refer to :obj:`numpy.ndarray.real`.

        Examples
        --------
        >>> import dpnp as np
        >>> x = np.sqrt(np.array([1+0j, 0+1j]))
        >>> x.real
        array([1.        , 0.70710677])

        """
        return dpnp.real(self)

    @real.setter
    def real(self, value):
        """
        Set the real part of the array.

        For full documentation refer to :obj:`numpy.ndarray.real`.

        Examples
        --------
        >>> import dpnp as np
        >>> a = np.array([1+2j, 3+4j, 5+6j])
        >>> a.real = 9
        >>> a
        array([9.+2.j, 9.+4.j, 9.+6.j])

        """
        dpnp.copyto(self._array_obj.real, value)

    def repeat(self, repeats, axis=None):
        """
        Repeat elements of an array.

        For full documentation refer to :obj:`dpnp.repeat`.

        """

        return dpnp.repeat(self, repeats, axis=axis)

    def reshape(self, *sh, **kwargs):
        """
        Returns an array containing the same data with a new shape.

        For full documentation refer to :obj:`numpy.ndarray.reshape`.

        Returns
        -------
        y : dpnp.ndarray
            This will be a new view object if possible;
            otherwise, it will be a copy.

        See Also
        --------
        :obj:`dpnp.reshape` : Equivalent function.

        Notes
        -----
        Unlike the free function `dpnp.reshape`, this method on `ndarray` allows
        the elements of the shape parameter to be passed in as separate arguments.
        For example, ``a.reshape(10, 11)`` is equivalent to
        ``a.reshape((10, 11))``.

        """

        if len(sh) == 1:
            sh = sh[0]
        return dpnp.reshape(self, sh, **kwargs)

    # 'resize',

    def round(self, decimals=0, out=None):
        """
        Return array with each element rounded to the given number of decimals.

        .. seealso::
           :obj:`dpnp.around` for full documentation.

        """

        return dpnp.around(self, decimals, out)

    # 'searchsorted',
    # 'setfield',
    # 'setflags',

    @property
    def shape(self):
        """Lengths of axes. A tuple of numbers represents size of each dimension.

        Setter of this property involves reshaping without copy. If the array
        cannot be reshaped without copy, it raises an exception.

        .. seealso: :attr:`numpy.ndarray.shape`

        """

        return self._array_obj.shape

    @shape.setter
    def shape(self, newshape):
        """
        Set new lengths of axes.

        A tuple of numbers represents size of each dimension.
        It involves reshaping without copy. If the array cannot be reshaped without copy,
        it raises an exception.

        .. seealso: :attr:`numpy.ndarray.shape`

        """

        self._array_obj.shape = newshape

    @property
    def size(self):
        """Number of elements in the array."""

        return self._array_obj.size

    def sort(self, axis=-1, kind=None, order=None):
        """
        Sort an array in-place.

        Refer to :obj:`dpnp.sort` for full documentation.

        Note
        ----
        `axis` in :obj:`dpnp.sort` could be integr or ``None``. If ``None``,
        the array is flattened before sorting. However, `axis` in :obj:`dpnp.ndarray.sort`
        can only be integer since it sorts an array in-place.

        Examples
        --------
        >>> import dpnp as np
        >>> a = np.array([[1,4],[3,1]])
        >>> a.sort(axis=1)
        >>> a
        array([[1, 4],
              [1, 3]])
        >>> a.sort(axis=0)
        >>> a
        array([[1, 1],
              [3, 4]])

        """

        if axis is None:
            raise TypeError(
                "'NoneType' object cannot be interpreted as an integer"
            )
        self[...] = dpnp.sort(self, axis=axis, kind=kind, order=order)

    def squeeze(self, axis=None):
        """
        Remove single-dimensional entries from the shape of an array.

        Refer to :obj:`dpnp.squeeze` for full documentation

        """

        return dpnp.squeeze(self, axis)

    def std(
        self,
        axis=None,
        dtype=None,
        out=None,
        ddof=0,
        keepdims=False,
        *,
        where=True,
    ):
        """
        Returns the standard deviation of the array elements, along given axis.

        Refer to :obj:`dpnp.std` for full documentation.
        """

        return dpnp.std(self, axis, dtype, out, ddof, keepdims, where=where)

    @property
    def strides(self):
        """
        Get strides of an array.

        Returns memory displacement in array elements, upon unit
        change of respective index.

        E.g. for strides (s1, s2, s3) and multi-index (i1, i2, i3)

           a[i1, i2, i3] == (&a[0,0,0])[ s1*s1 + s2*i2 + s3*i3]

        """

        return self._array_obj.strides

    def sum(
        self,
        /,
        *,
        axis=None,
        dtype=None,
        keepdims=False,
        out=None,
        initial=0,
        where=True,
    ):
        """
        Returns the sum along a given axis.

        For full documentation refer to :obj:`dpnp.sum`.
        """

        return dpnp.sum(
            self,
            axis=axis,
            dtype=dtype,
            out=out,
            keepdims=keepdims,
            initial=initial,
            where=where,
        )

    def swapaxes(self, axis1, axis2):
        """
        Interchange two axes of an array.

        For full documentation refer to :obj:`numpy.swapaxes`.
        """

        return dpnp.swapaxes(self, axis1=axis1, axis2=axis2)

    def take(self, indices, /, *, axis=None, out=None, mode="wrap"):
        """
        Take elements from an array along an axis.

        For full documentation refer to :obj:`numpy.take`.

        """

        return dpnp.take(self, indices, axis=axis, out=out, mode=mode)

    # 'tobytes',
    # 'tofile',
    # 'tolist',
    # 'tostring',
    # 'trace',

    def transpose(self, *axes):
        """
        Returns a view of the array with axes transposed.

        For full documentation refer to :obj:`numpy.ndarray.transpose`.

        Returns
        -------
        y : dpnp.ndarray
            View of the array with its axes suitably permuted.

        See Also
        --------
            :obj:`dpnp.transpose` : Equivalent function.
            :obj:`dpnp.ndarray.ndarray.T` : Array property returning the array transposed.
            :obj:`dpnp.ndarray.reshape` : Give a new shape to an array without changing its data.

        Examples
        --------
        >>> import dpnp as dp
        >>> a = dp.array([[1, 2], [3, 4]])
        >>> a
        array([[1, 2],
               [3, 4]])
        >>> a.transpose()
        array([[1, 3],
               [2, 4]])
        >>> a.transpose((1, 0))
        array([[1, 3],
               [2, 4]])

        >>> a = dp.array([1, 2, 3, 4])
        >>> a
        array([1, 2, 3, 4])
        >>> a.transpose()
        array([1, 2, 3, 4])

        """

        ndim = self.ndim
        if ndim < 2:
            return self

        axes_len = len(axes)
        if axes_len == 1 and isinstance(axes[0], tuple):
            axes = axes[0]

        res = self.__new__(dpnp_array)
        if ndim == 2 and axes_len == 0:
            res._array_obj = self._array_obj.T
        else:
            if len(axes) == 0 or axes[0] is None:
                # self.transpose().shape == self.shape[::-1]
                # self.transpose(None).shape == self.shape[::-1]
                axes = tuple((ndim - x - 1) for x in range(ndim))

            res._array_obj = dpt.permute_dims(self._array_obj, axes)
        return res

    def var(
        self,
        axis=None,
        dtype=None,
        out=None,
        ddof=0,
        keepdims=False,
        *,
        where=True,
    ):
        """
        Returns the variance of the array elements, along given axis.

        Refer to :obj:`dpnp.var` for full documentation.
        """
        return dpnp.var(self, axis, dtype, out, ddof, keepdims, where=where)


# 'view'
