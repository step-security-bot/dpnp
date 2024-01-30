import operator
import unittest

import numpy
import pytest
import warnings
import dpnp
from tests.third_party.cupy import testing
from tests.helper import has_support_aspect64

@testing.parameterize(
    *testing.product(
        {
            "shape_pair": [
                # dot test
                ((3, 2), (2, 4)),
                ((3, 0), (0, 4)),
                ((0, 2), (2, 4)),
                ((3, 2), (2, 0)),
                ((2,), (2, 4)),
                ((0,), (0, 4)),
                ((3, 2), (2,)),
                ((3, 0), (0,)),
                ((2,), (2,)),
                ((0,), (0,)),
                # matmul test
                ((5, 3, 2), (5, 2, 4)),
                ((0, 3, 2), (0, 2, 4)),
                ((5, 3, 2), (2, 4)),
                ((0, 3, 2), (2, 4)),
                ((3, 2), (5, 2, 4)),
                ((3, 2), (0, 2, 4)),
                ((5, 3, 2), (1, 2, 4)),
                ((0, 3, 2), (1, 2, 4)),
                ((1, 3, 2), (5, 2, 4)),
                ((1, 3, 2), (0, 2, 4)),
                ((5, 3, 2), (2,)),
                ((5, 3, 0), (0,)),
                ((2,), (5, 2, 4)),
                ((0,), (5, 0, 4)),
                ((2, 2, 3, 2), (2, 2, 2, 4)),
                ((5, 0, 3, 2), (5, 0, 2, 4)),
                ((6, 5, 3, 2), (2, 4)),
                ((5, 0, 3, 2), (2, 4)),
                ((3, 2), (6, 5, 2, 4)),
                ((3, 2), (5, 0, 2, 4)),
                ((1, 5, 3, 2), (6, 1, 2, 4)),
                ((1, 0, 3, 2), (6, 1, 2, 4)),
                ((6, 1, 3, 2), (1, 5, 2, 4)),
                ((6, 1, 3, 2), (1, 0, 2, 4)),
                ((6, 5, 3, 2), (2,)),
                ((6, 5, 3, 0), (0,)),
                ((2,), (6, 5, 2, 4)),
                ((0,), (6, 5, 0, 4)),
                ((1, 3, 3), (10, 1, 3, 1)),
            ],
        }
    )
)
class TestMatmul(unittest.TestCase):
    @testing.for_all_dtypes(name="dtype1")
    @testing.numpy_cupy_allclose(rtol=1e-3, atol=1e-3)  # required for uint8
    def test_operator_matmul(self, xp, dtype1):
        x1 = testing.shaped_arange(self.shape_pair[0], xp, dtype1)
        x2 = testing.shaped_arange(self.shape_pair[1], xp, dtype1)
        return operator.matmul(x1, x2)

    @testing.for_all_dtypes(name="dtype1")
    @testing.numpy_cupy_allclose(rtol=1e-3, atol=1e-3)  # required for uint8
    def test_cupy_matmul(self, xp, dtype1):
        x1 = testing.shaped_arange(self.shape_pair[0], xp, dtype1)
        x2 = testing.shaped_arange(self.shape_pair[1], xp, dtype1)
        return xp.matmul(x1, x2)


@testing.parameterize(
    *testing.product(
        {
            "shape_pair": [
                ((6, 5, 3, 2), (6, 5, 2, 4)),
                ((6, 5, 3, 2), (6, 1, 2, 4)),
                ((6, 5, 3, 2), (1, 5, 2, 4)),
                ((6, 5, 3, 2), (1, 1, 2, 4)),
                ((6, 1, 3, 2), (6, 5, 2, 4)),
                ((1, 5, 3, 2), (6, 5, 2, 4)),
                ((1, 1, 3, 2), (6, 5, 2, 4)),
                ((3, 2), (6, 5, 2, 4)),
                ((6, 5, 3, 2), (2, 4)),
                ((2,), (6, 5, 2, 4)),
                ((6, 5, 3, 2), (2,)),
            ],
        }
    )
)
class TestMatmulLarge(unittest.TestCase):
    # Avoid overflow
    skip_dtypes = {
        (numpy.int8, numpy.uint8),
        (numpy.int8, numpy.int16),
        (numpy.int8, numpy.float16),
        (numpy.uint8, numpy.uint8),
        (numpy.uint8, numpy.int16),
        (numpy.uint8, numpy.uint16),
        (numpy.int16, numpy.int16),
        (numpy.uint16, numpy.uint16),
    }

    @testing.for_all_dtypes(name="dtype1")
    @testing.numpy_cupy_allclose(rtol=1e-3, atol=1e-3)  # required for uint8
    def test_operator_matmul(self, xp, dtype1):
        if (dtype1, dtype1) in self.skip_dtypes or (
            dtype1,
            dtype1,
        ) in self.skip_dtypes:
            return xp.array([])
        x1 = testing.shaped_random(self.shape_pair[0], xp, dtype1)
        x2 = testing.shaped_random(self.shape_pair[1], xp, dtype1)
        return operator.matmul(x1, x2)

    @testing.for_all_dtypes(name="dtype1")
    @testing.numpy_cupy_allclose(rtol=1e-3, atol=1e-3)  # required for uint8
    def test_cupy_matmul(self, xp, dtype1):
        if (dtype1, dtype1) in self.skip_dtypes or (
            dtype1,
            dtype1,
        ) in self.skip_dtypes:
            return xp.array([])
        shape1, shape2 = self.shape_pair
        x1 = testing.shaped_random(shape1, xp, dtype1)
        x2 = testing.shaped_random(shape2, xp, dtype1)
        return xp.matmul(x1, x2)


class TestMatmulStrides:
    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(rtol=1e-3, atol=1e-3)  # required for uint8
    def test_relaxed_c_contiguous_input(self, xp, dtype):
        x1 = testing.shaped_arange((2, 2, 3), xp, dtype)[:, None, :, :]
        x2 = testing.shaped_arange((2, 1, 3, 1), xp, dtype)
        return x1 @ x2


# Define constants for compute types
COMPUTE_TYPE_TBD = 0
COMPUTE_TYPE_DEFAULT = 1
COMPUTE_TYPE_PEDANTIC = 2
COMPUTE_TYPE_FP16 = 3
COMPUTE_TYPE_FP32 = 4
COMPUTE_TYPE_FP64 = 5
COMPUTE_TYPE_BF16 = 6
COMPUTE_TYPE_TF32 = 7

# Define compute type list and dictionary
compute_types = [COMPUTE_TYPE_TBD, COMPUTE_TYPE_TBD, COMPUTE_TYPE_TBD]
compute_type_str = {
    0: 'COMPUTE_TYPE_TBD',
    1: 'COMPUTE_TYPE_DEFAULT',
    2: 'COMPUTE_TYPE_PEDANTIC',
    3: 'COMPUTE_TYPE_FP16',
    4: 'COMPUTE_TYPE_FP32',
    5: 'COMPUTE_TYPE_FP64',
    6: 'COMPUTE_TYPE_BF16',
    7: 'COMPUTE_TYPE_TF32',
}

# Function to compute the index for the given dtype
def to_compute_type_index(dtype):
    dtype_char = numpy.dtype(dtype).char
    if dtype_char == 'e':
        return 0
    elif dtype_char in 'fF':
        return 1
    elif dtype_char in 'dD':
        return 2
    else:
        raise TypeError('dtype is not supported: {}'.format(dtype))

# Function to set compute type
def _set_compute_type(dtype, compute_type):
    global compute_types
    if compute_type in (COMPUTE_TYPE_TBD, COMPUTE_TYPE_DEFAULT,
                        COMPUTE_TYPE_PEDANTIC, COMPUTE_TYPE_FP16,
                        COMPUTE_TYPE_FP32, COMPUTE_TYPE_FP64):
        compute_types[to_compute_type_index(dtype)] = compute_type
    elif compute_type in (COMPUTE_TYPE_BF16, COMPUTE_TYPE_TF32):
        #if int(cupy.cuda.Device().compute_capability()[0]) >= 8:
        #    compute_types[to_compute_type_index(dtype)] = compute_type
        #else:
        warnings.warn('COMPUTE_TYPE_BF16 and COMPUTE_TYPE_TF32 are only '
                        'available on GPUs with compute capability 8.0 or '
                        'higher. COMPUTE_TYPE_DEFAULT will be used instead.')
        compute_types[to_compute_type_index(dtype)] = COMPUTE_TYPE_DEFAULT
    else:
        raise ValueError('Unknown compute type: {}'.format(compute_type))

# Function to get compute type
def _get_compute_type(dtype):
    global compute_types
    index = to_compute_type_index(dtype)
    if compute_types[index] == COMPUTE_TYPE_TBD:
        compute_type = COMPUTE_TYPE_DEFAULT
        dtype_char = numpy.dtype(dtype).char
        #if dtype_char in 'fF' and int(os.getenv('CUPY_TF32', '0')) > 0:
        #    compute_type = COMPUTE_TYPE_TF32
        _set_compute_type(dtype, compute_type)
    return compute_types[index]

class _TestMatmulComputeTypes(unittest.TestCase):
    def setUp(self):
        self.old_compute_type = _get_compute_type(self.dtype)
        _set_compute_type(self.dtype, self.compute_type)

    def tearDown(self):
        _set_compute_type(self.dtype, self.old_compute_type)

    def make_x1_x2(self, xp, shapes, dtypes):
        x1 = testing.shaped_random(shapes[0], xp, dtypes[0])
        x2 = testing.shaped_random(shapes[1], xp, dtypes[1])
        return x1, x2

@testing.parameterize(
    *testing.product({
        'compute_type': [
            COMPUTE_TYPE_DEFAULT,
        #    COMPUTE_TYPE_PEDANTIC,
        ],
        'shape_pair': [
            ((32, 64), (64, 96)),
           # ((64, 96), (96, 32)),
           # ((96, 32), (32, 64)),
        ],
    }))
class TestMatmulFp16ComputeTypes(_TestMatmulComputeTypes):
    dtype = numpy.float16

    @testing.numpy_cupy_allclose(rtol=1e-3, atol=1e-3)
    def test_operator_matmul(self, xp):
        x1, x2 = self.make_x1_x2(xp, self.shape_pair, (self.dtype, self.dtype))
        return operator.matmul(x1, x2)

    @testing.numpy_cupy_allclose(rtol=1e-3, atol=1e-3)
    def test_cupy_matmul(self, xp):
        x1, x2 = self.make_x1_x2(xp, self.shape_pair, (self.dtype, self.dtype))
        return xp.matmul(x1, x2)
    

@testing.parameterize(
    *testing.product({
        'compute_type': [
            COMPUTE_TYPE_DEFAULT,
            COMPUTE_TYPE_PEDANTIC,
            COMPUTE_TYPE_TF32,
        ],
        'shape_pair': [
            ((100, 200), (200, 300)),
            ((200, 300), (300, 100)),
            ((300, 100), (100, 200)),
        ],
        'dtype_pair': [
            (numpy.float16, numpy.float32),
            (numpy.float32, numpy.float32),
            (numpy.float16, numpy.complex64),
            (numpy.float32, numpy.complex64),
            (numpy.complex64, numpy.complex64),
        ],
    }))
class TestMatmulFp32ComputeTypes(_TestMatmulComputeTypes):
    dtype = numpy.float32

    @testing.numpy_cupy_allclose(rtol=1e-3, atol=1e-3)
    def test_operator_matmul(self, xp):
        x1, x2 = self.make_x1_x2(xp, self.shape_pair, self.dtype_pair)
        return operator.matmul(x1, x2)

    @testing.numpy_cupy_allclose(rtol=1e-3, atol=1e-3)
    def test_cupy_matmul(self, xp):
        x1, x2 = self.make_x1_x2(xp, self.shape_pair, self.dtype_pair)
        return xp.matmul(x1, x2)


@testing.parameterize(
    *testing.product({
        'compute_type': [
            COMPUTE_TYPE_DEFAULT,
            COMPUTE_TYPE_PEDANTIC,
        ],
        'shape_pair': [
            ((100, 200), (200, 300)),
            ((200, 300), (300, 100)),
            ((300, 100), (100, 200)),
        ],
        'dtype_pair': [
            (numpy.float32, numpy.float64),
            (numpy.float64, numpy.float64),
            (numpy.float32, numpy.complex128),
            (numpy.float64, numpy.complex128),
            (numpy.complex64, numpy.complex128),
            (numpy.complex128, numpy.complex128),
        ],
    }))
@pytest.mark.skipif(not has_support_aspect64(), reason="No fp64 support by device")
class TestMatmulFp64ComputeTypes(_TestMatmulComputeTypes):
    dtype = numpy.float64

    @testing.numpy_cupy_allclose()
    def test_operator_matmul(self, xp):
        x1, x2 = self.make_x1_x2(xp, self.shape_pair, self.dtype_pair)
        return operator.matmul(x1, x2)

    @testing.numpy_cupy_allclose()
    def test_cupy_matmul(self, xp):
        x1, x2 = self.make_x1_x2(xp, self.shape_pair, self.dtype_pair)
        return xp.matmul(x1, x2)
    

@testing.parameterize(
    *testing.product(
        {
            "shape_pair": [
                ((5, 3, 1), (3, 1, 4)),
                ((3, 2, 3), (3, 2, 4)),
                ((3, 2), ()),
                ((), (3, 2)),
                ((), ()),
                ((3, 2), (1,)),
                ((0, 2), (3, 0)),
                ((0, 1, 1), (2, 1, 1)),
            ],
        }
    )
)
class TestMatmulInvalidShape(unittest.TestCase):
    def test_invalid_shape(self):
        for xp in (numpy, dpnp):
            shape1, shape2 = self.shape_pair
            x1 = testing.shaped_arange(shape1, xp, numpy.float32)
            x2 = testing.shaped_arange(shape2, xp, numpy.float32)
            with pytest.raises(ValueError):
                xp.matmul(x1, x2)


@testing.parameterize(
    *testing.product({
        'shapes_axes': [
            (((2, 5, 3, 2, 3, 4),  (3, 5, 1, 1, 1, 4), (5, 5, 2, 2, 3, 4)),
             [(1, 2), (0, 1), (0, 1)]),
            (((2, 5, 3, 2, 3, 4),  (2, 5, 3, 1, 4, 1), (3, 1, 2, 5, 3, 2)),
             [(-2, -1), (-2, -1), (0, 1)]),
            (((3, 2, 4, 4), (4, 4, 3, 2), (4, 4, 3, 3)),
             [(0, 1), (-1, -2), (-2, -1)]),
            (((3, 2, 4, 4), (2, 3, 4, 4), (4, 3, 3, 4)),
             [(0, 1), (0, 1), (1, 2)]),
        ],
    }))
class TestMatmulAxes(unittest.TestCase):
    @testing.numpy_cupy_allclose(rtol=1e-3, atol=1e-3)  # required for uint8
    def test_cupy_matmul_axes(self, xp):
        x1 = testing.shaped_arange(self.shapes_axes[0][0], xp)
        x2 = testing.shaped_arange(self.shapes_axes[0][1], xp)
        return xp.matmul(x1, x2, axes=self.shapes_axes[1])

    @testing.numpy_cupy_allclose(rtol=1e-3, atol=1e-3, type_check=has_support_aspect64())  # required for uint8
    def test_cupy_matmul_axes_out(self, xp):
        x1 = testing.shaped_arange(self.shapes_axes[0][0], xp)
        x2 = testing.shaped_arange(self.shapes_axes[0][1], xp)
        out = xp.zeros(self.shapes_axes[0][2])
        result = xp.matmul(x1, x2, axes=self.shapes_axes[1], out=out)
        assert out is result
        return out 
    
#class TestMatmulDispatch(unittest.TestCase):
#    def test_matmul_dispatch(self):
#        x1 = testing.shaped_arange((2, 10, 5), cupy)
#        x2 = testing.shaped_arange((10, 2, 5), cupy)
#        o_np = numpy.matmul(x1, x2, axes=[(0, 1), (0, 1), (0, 1)])
#        assert isinstance(o_np, cupy.ndarray)
#        o_cp = cupy.matmul(x1, x2, axes=[(0, 1), (0, 1), (0, 1)])
#        testing.assert_allclose(o_np, o_cp)    
    
class TestMatmulOverflow(unittest.TestCase):
    @testing.for_int_dtypes(name='dtype', no_bool=True)
    @testing.numpy_cupy_allclose(rtol=1e-3, atol=1e-3)  # required for uint8
    def test_overflow(self, xp, dtype):
        value = numpy.iinfo(dtype).max
        print("value", value)
        a = xp.array([value - 10]).astype(dtype)
        b = xp.array([value - 10]).astype(dtype)
        return xp.matmul(a, b)    