import itertools
import unittest
import warnings

import numpy
import pytest

import dpnp as cupy
from tests.helper import has_support_aspect64
from tests.third_party.cupy import testing

float_types = list(testing.helper._float_dtypes)
complex_types = []
signed_int_types = [numpy.int32, numpy.int64]
unsigned_int_types = []
int_types = signed_int_types + unsigned_int_types
all_types = float_types + int_types + complex_types
negative_types_wo_fp16 = (
    [numpy.bool_]
    + [numpy.float32, numpy.float64]
    + [numpy.int16, numpy.int32, numpy.int64]
    + complex_types
)
negative_types = float_types + signed_int_types + complex_types
negative_no_complex_types = float_types + signed_int_types
no_complex_types = float_types + int_types


@testing.parameterize(
    *(
        testing.product(
            {
                "nargs": [1],
                "name": [
                    "reciprocal",
                    "conj",
                    "conjugate",
                    "angle",
                ],
            }
        )
        + testing.product(
            {
                "nargs": [2],
                "name": [
                    "add",
                    "multiply",
                    "divide",
                    "power",
                    "subtract",
                    "true_divide",
                    "floor_divide",
                    "fmod",
                    "remainder",
                    "mod",
                ],
            }
        )
    )
)
class TestArithmeticRaisesWithNumpyInput:
    def test_raises_with_numpy_input(self):
        nargs = self.nargs
        name = self.name

        # Check TypeError is raised if numpy.ndarray is given as input
        func = getattr(cupy, name)
        for input_xp_list in itertools.product(*[[numpy, cupy]] * nargs):
            if all(xp is cupy for xp in input_xp_list):
                # We don't test all-cupy-array inputs here
                continue
            arys = [xp.array([2, -3]) for xp in input_xp_list]
            with pytest.raises(TypeError):
                func(*arys)


@testing.parameterize(
    *(
        testing.product(
            {
                "arg1": (
                    [
                        testing.shaped_arange((2, 3), numpy, dtype=d)
                        for d in all_types
                    ]
                ),
                "name": ["conj", "conjugate", "real", "imag"],
            }
        )
        + testing.product(
            {
                "arg1": (
                    [
                        testing.shaped_arange((2, 3), numpy, dtype=d)
                        for d in all_types
                    ]
                ),
                "deg": [True, False],
                "name": ["angle"],
            }
        )
        + testing.product(
            {
                "arg1": (
                    [
                        numpy.array([-3, -2, -1, 1, 2, 3], dtype=d)
                        for d in negative_types_wo_fp16
                    ]
                ),
                "deg": [True, False],
                "name": ["angle"],
            }
        )
        + testing.product(
            {
                "arg1": (
                    [
                        testing.shaped_arange((2, 3), numpy, dtype=d) + 1
                        for d in all_types
                    ]
                ),
                "name": ["reciprocal"],
            }
        )
    )
)
class TestArithmeticUnary:
    @testing.numpy_cupy_allclose(atol=1e-5, type_check=has_support_aspect64())
    def test_unary(self, xp):
        arg1 = self.arg1
        if isinstance(arg1, numpy.ndarray):
            arg1 = xp.asarray(arg1)

        if self.name in ("reciprocal") and xp is numpy:
            # In Numpy, for integer arguments with absolute value larger than 1 the result is always zero.
            # We need to convert the input data type to float then compare the output with DPNP.
            if isinstance(arg1, numpy.ndarray) and numpy.issubdtype(
                arg1.dtype, numpy.integer
            ):
                np_dtype = (
                    numpy.float64 if has_support_aspect64() else numpy.float32
                )
                arg1 = xp.asarray(arg1, dtype=np_dtype)

        if self.name in {"angle"}:
            y = getattr(xp, self.name)(arg1, self.deg)
            # In Numpy, for boolean arguments the output data type is always default floating data type.
            # while data type of output in DPNP is determined by Type Promotion Rules.
            if (
                isinstance(arg1, cupy.ndarray)
                and cupy.issubdtype(arg1.dtype, cupy.bool)
                and has_support_aspect64()
            ):
                y = y.astype(cupy.float64)
        else:
            y = getattr(xp, self.name)(arg1)

        # if self.name in ("real", "imag"):
        # Some NumPy functions return Python scalars for Python scalar
        # inputs.
        # We need to convert them to arrays to compare with CuPy outputs.
        # if xp is numpy and isinstance(arg1, (bool, int, float, complex)):
        #    y = xp.asarray(y)

        # TODO(niboshi): Fix this
        # numpy.real and numpy.imag return Python int if the input is
        # Python bool. CuPy should return an array of dtype.int32 or
        # dtype.int64 (depending on the platform) in such cases, instead
        # of an array of dtype.bool.
        # if xp is cupy and isinstance(arg1, bool):
        #    y = y.astype(int)

        return y


@testing.parameterize(
    *testing.product(
        {
            "shape": [(3, 2), (), (3, 0, 2)],
        }
    )
)
class TestComplex:
    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_cupy_array_equal()
    def test_real_ndarray_nocomplex(self, xp, dtype):
        x = testing.shaped_arange(self.shape, xp, dtype=dtype)
        real = x.real
        assert real is x  # real returns self
        return real

    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_cupy_array_equal()
    def test_real_nocomplex(self, xp, dtype):
        x = testing.shaped_arange(self.shape, xp, dtype=dtype)
        real = xp.real(x)
        assert real is x  # real returns self
        return real

    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_cupy_array_equal()
    def test_imag_ndarray_nocomplex(self, xp, dtype):
        x = testing.shaped_arange(self.shape, xp, dtype=dtype)
        imag = x.imag
        return imag

    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_cupy_array_equal()
    def test_imag_nocomplex(self, xp, dtype):
        x = testing.shaped_arange(self.shape, xp, dtype=dtype)
        imag = xp.imag(x)
        return imag


class ArithmeticBinaryBase:
    @testing.numpy_cupy_allclose(atol=1e-4, type_check=False)
    def check_binary(self, xp):
        arg1 = self.arg1
        arg2 = self.arg2
        np1 = numpy.asarray(arg1)
        np2 = numpy.asarray(arg2)
        dtype1 = np1.dtype
        dtype2 = np2.dtype

        # TODO(niboshi): Fix this: xp.add(0j, xp.array([2.], 'f')).dtype
        #     numpy => complex64
        #     cupy => complex128
        if isinstance(arg1, complex):
            if dtype2 in (numpy.float16, numpy.float32):
                return xp.array(True)

        if isinstance(arg1, numpy.ndarray):
            arg1 = xp.asarray(arg1)
        if isinstance(arg2, numpy.ndarray):
            arg2 = xp.asarray(arg2)

        # Subtraction between booleans is not allowed.
        if (
            self.name == "subtract"
            and dtype1 == numpy.bool_
            and dtype2 == numpy.bool_
        ):
            return xp.array(True)

        func = getattr(xp, self.name)
        with testing.NumpyError(divide="ignore"):
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                if self.use_dtype:
                    y = func(arg1, arg2, dtype=self.dtype)
                else:
                    y = func(arg1, arg2)

        # TODO(niboshi): Fix this. If rhs is a Python complex,
        #    numpy returns complex64
        #    cupy returns complex128
        if xp is cupy and isinstance(arg2, complex):
            if dtype1 in (numpy.float16, numpy.float32):
                y = y.astype(numpy.complex64)

        # NumPy returns an output array of another type than DPNP when input ones have different types.
        if xp is numpy and dtype1 != dtype2:
            is_array_arg1 = not xp.isscalar(arg1)
            is_array_arg2 = not xp.isscalar(arg2)

            is_int_float = lambda _x, _y: numpy.issubdtype(
                _x, numpy.integer
            ) and numpy.issubdtype(_y, numpy.floating)

        return y


@testing.gpu
@testing.parameterize(
    *(
        testing.product(
            {
                "arg1": [
                    testing.shaped_arange((2, 3), numpy, dtype=d)
                    for d in all_types
                ]
                + [0, 0.0, 2, 2.0],
                "arg2": [
                    testing.shaped_reverse_arange((2, 3), numpy, dtype=d)
                    for d in all_types
                ]
                + [0, 0.0, 2, 2.0],
                "name": ["add", "multiply", "power", "subtract"],
            }
        )
        + testing.product(
            {
                "arg1": [
                    numpy.array([-3, -2, -1, 1, 2, 3], dtype=d)
                    for d in negative_types
                ]
                + [0, 0.0, 2, 2.0, -2, -2.0],
                "arg2": [
                    numpy.array([-3, -2, -1, 1, 2, 3], dtype=d)
                    for d in negative_types
                ]
                + [0, 0.0, 2, 2.0, -2, -2.0],
                "name": ["divide", "true_divide", "subtract"],
            }
        )
    )
)
@pytest.mark.usefixtures("allow_fall_back_on_numpy")
class TestArithmeticBinary(ArithmeticBinaryBase, unittest.TestCase):
    def test_binary(self):
        self.use_dtype = False
        self.check_binary()


@testing.gpu
@testing.parameterize(
    *(
        testing.product(
            {
                "arg1": [
                    numpy.array([-3, -2, -1, 1, 2, 3], dtype=d)
                    for d in int_types
                ]
                + [0, 0.0, 2, 2.0, -2, -2.0],
                "arg2": [
                    numpy.array([-3, -2, -1, 1, 2, 3], dtype=d)
                    for d in int_types
                ]
                + [0, 0.0, 2, 2.0, -2, -2.0],
                "name": ["true_divide"],
                "dtype": [numpy.float64],
                "use_dtype": [True, False],
            }
        )
        + testing.product(
            {
                "arg1": [
                    numpy.array([-3, -2, -1, 1, 2, 3], dtype=d)
                    for d in float_types
                ]
                + [0.0, 2.0, -2.0],
                "arg2": [
                    numpy.array([-3, -2, -1, 1, 2, 3], dtype=d)
                    for d in float_types
                ]
                + [0.0, 2.0, -2.0],
                "name": ["power", "true_divide", "subtract"],
                "dtype": [numpy.float64],
                "use_dtype": [True, False],
            }
        )
        + testing.product(
            {
                "arg1": [
                    testing.shaped_arange((2, 3), numpy, dtype=d)
                    for d in no_complex_types
                ]
                + [0, 0.0, 2, 2.0, -2, -2.0],
                "arg2": [
                    testing.shaped_reverse_arange((2, 3), numpy, dtype=d)
                    for d in no_complex_types
                ]
                + [0, 0.0, 2, 2.0, -2, -2.0],
                "name": ["floor_divide", "fmod", "remainder", "mod"],
                "dtype": [numpy.float64],
                "use_dtype": [True, False],
            }
        )
        + testing.product(
            {
                "arg1": [
                    numpy.array([-3, -2, -1, 1, 2, 3], dtype=d)
                    for d in negative_no_complex_types
                ]
                + [0, 0.0, 2, 2.0, -2, -2.0],
                "arg2": [
                    numpy.array([-3, -2, -1, 1, 2, 3], dtype=d)
                    for d in negative_no_complex_types
                ]
                + [0, 0.0, 2, 2.0, -2, -2.0],
                "name": ["floor_divide", "fmod", "remainder", "mod"],
                "dtype": [numpy.float64],
                "use_dtype": [True, False],
            }
        )
    )
)
@pytest.mark.usefixtures("allow_fall_back_on_numpy")
class TestArithmeticBinary2(ArithmeticBinaryBase, unittest.TestCase):
    def test_binary(self):
        if (
            self.use_dtype
            and numpy.lib.NumpyVersion(numpy.__version__) < "1.10.0"
        ):
            raise unittest.SkipTest("NumPy>=1.10")
        self.check_binary()


class TestArithmeticModf(unittest.TestCase):
    @testing.for_float_dtypes()
    @testing.numpy_cupy_allclose()
    def test_modf(self, xp, dtype):
        a = xp.array([-2.5, -1.5, -0.5, 0, 0.5, 1.5, 2.5], dtype=dtype)
        b, c = xp.modf(a)
        d = xp.empty((2, 7), dtype=dtype)
        d[0] = b
        d[1] = c
        return d


@testing.parameterize(
    *testing.product({"xp": [numpy, cupy], "shape": [(3, 2), (), (3, 0, 2)]})
)
@testing.gpu
class TestBoolSubtract(unittest.TestCase):
    def test_bool_subtract(self):
        xp = self.xp
        if xp is numpy and not testing.numpy_satisfies(">=1.14.0"):
            raise unittest.SkipTest("NumPy<1.14.0")
        shape = self.shape
        x = testing.shaped_random(shape, xp, dtype=numpy.bool_)
        y = testing.shaped_random(shape, xp, dtype=numpy.bool_)
        with pytest.raises(TypeError):
            xp.subtract(x, y)
