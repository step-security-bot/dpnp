import sys
import unittest

import numpy
import pytest
from dpctl import select_default_device

import dpnp as cupy
from tests.helper import has_support_aspect64
from tests.third_party.cupy import testing


@testing.gpu
class TestCorrcoef(unittest.TestCase):
    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_corrcoef(self, xp, dtype):
        a = testing.shaped_arange((2, 3), xp, dtype)
        return xp.corrcoef(a)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_corrcoef_diag_exception(self, xp, dtype):
        a = testing.shaped_arange((1, 3), xp, dtype)
        return xp.corrcoef(a)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_corrcoef_y(self, xp, dtype):
        a = testing.shaped_arange((2, 3), xp, dtype)
        y = testing.shaped_arange((2, 3), xp, dtype)
        return xp.corrcoef(a, y=y)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_corrcoef_rowvar(self, xp, dtype):
        a = testing.shaped_arange((2, 3), xp, dtype)
        y = testing.shaped_arange((2, 3), xp, dtype)
        return xp.corrcoef(a, y=y, rowvar=False)


class TestCov(unittest.TestCase):
    # resulting dtype will differ with numpy if no fp64 support by a default device
    _has_fp64 = select_default_device().has_aspect_fp64

    def generate_input(self, a_shape, y_shape, xp, dtype):
        a = testing.shaped_arange(a_shape, xp, dtype)
        y = None
        if y_shape is not None:
            y = testing.shaped_arange(y_shape, xp, dtype)
        return a, y

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(type_check=_has_fp64, accept_error=True)
    def check(
        self,
        a_shape,
        y_shape=None,
        rowvar=True,
        bias=False,
        ddof=None,
        xp=None,
        dtype=None,
        fweights=None,
        aweights=None,
        name=None,
    ):
        a, y = self.generate_input(a_shape, y_shape, xp, dtype)
        if fweights is not None:
            fweights = name.asarray(fweights)
        if aweights is not None:
            aweights = name.asarray(aweights)
        # print(type(fweights))
        # return xp.cov(a, y, rowvar, bias, ddof,
        #               fweights, aweights, dtype=dtype)
        return xp.cov(a, y, rowvar, bias, ddof, fweights, aweights)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(accept_error=True)
    def check_warns(
        self,
        a_shape,
        y_shape=None,
        rowvar=True,
        bias=False,
        ddof=None,
        xp=None,
        dtype=None,
        fweights=None,
        aweights=None,
    ):
        with testing.assert_warns(RuntimeWarning):
            a, y = self.generate_input(a_shape, y_shape, xp, dtype)
            return xp.cov(
                a, y, rowvar, bias, ddof, fweights, aweights, dtype=dtype
            )

    @testing.for_all_dtypes()
    def check_raises(
        self,
        a_shape,
        y_shape=None,
        rowvar=True,
        bias=False,
        ddof=None,
        dtype=None,
        fweights=None,
        aweights=None,
    ):
        for xp in (numpy, cupy):
            a, y = self.generate_input(a_shape, y_shape, xp, dtype)
            with pytest.raises(ValueError):
                xp.cov(
                    a, y, rowvar, bias, ddof, fweights, aweights, dtype=dtype
                )

    @pytest.mark.usefixtures("allow_fall_back_on_numpy")
    def test_cov(self):
        self.check((2, 3))
        self.check((2,), (2,))
        self.check((1, 3), (1, 3), rowvar=False)
        self.check((2, 3), (2, 3), rowvar=False)
        self.check((2, 3), bias=True)
        self.check((2, 3), ddof=2)
        self.check((2, 3))
        self.check((1, 3), fweights=(1, 4, 1))
        self.check((1, 3), aweights=(1.0, 4.0, 1.0))
        self.check((1, 3), bias=True, aweights=(1.0, 4.0, 1.0))
        self.check((1, 3), fweights=(1, 4, 1), aweights=(1.0, 4.0, 1.0))

    @pytest.mark.usefixtures("allow_fall_back_on_numpy")
    def test_cov_warns(self):
        self.check_warns((2, 3), ddof=3)
        self.check_warns((2, 3), ddof=4)

    @pytest.mark.usefixtures("allow_fall_back_on_numpy")
    def test_cov_raises(self):
        self.check_raises((2, 3), ddof=1.2)
        self.check_raises((3, 4, 2))
        self.check_raises((2, 3), (3, 4, 2))

    def test_cov_empty(self):
        self.check((0, 1))


@testing.gpu
@testing.parameterize(
    *testing.product(
        {
            "mode": ["valid", "same", "full"],
            "shape1": [(5,), (6,), (20,), (21,)],
            "shape2": [(5,), (6,), (20,), (21,)],
        }
    )
)
@pytest.mark.usefixtures("allow_fall_back_on_numpy")
class TestCorrelateShapeCombination(unittest.TestCase):
    @testing.for_all_dtypes(no_float16=True)
    @testing.numpy_cupy_allclose(rtol=1e-4)
    def test_correlate(self, xp, dtype):
        a = testing.shaped_arange(self.shape1, xp, dtype)
        b = testing.shaped_arange(self.shape2, xp, dtype)
        return xp.correlate(a, b, mode=self.mode)


@testing.gpu
@testing.parameterize(*testing.product({"mode": ["valid", "full", "same"]}))
@pytest.mark.usefixtures("allow_fall_back_on_numpy")
class TestCorrelate(unittest.TestCase):
    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(rtol=1e-5)
    def test_correlate_non_contiguous(self, xp, dtype):
        a = testing.shaped_arange((300,), xp, dtype)
        b = testing.shaped_arange((100,), xp, dtype)
        return xp.correlate(a[::200], b[10::70], mode=self.mode)

    @testing.for_all_dtypes(no_float16=True)
    @testing.numpy_cupy_allclose(rtol=1e-4)
    def test_correlate_large_non_contiguous(self, xp, dtype):
        a = testing.shaped_arange((10000,), xp, dtype)
        b = testing.shaped_arange((1000,), xp, dtype)
        return xp.correlate(a[200::], b[10::700], mode=self.mode)

    @testing.for_all_dtypes_combination(names=["dtype1", "dtype2"])
    @testing.numpy_cupy_allclose(rtol=1e-2, type_check=has_support_aspect64())
    def test_correlate_diff_types(self, xp, dtype1, dtype2):
        a = testing.shaped_random((200,), xp, dtype1)
        b = testing.shaped_random((100,), xp, dtype2)
        return xp.correlate(a, b, mode=self.mode)


@testing.gpu
@testing.parameterize(*testing.product({"mode": ["valid", "same", "full"]}))
@pytest.mark.usefixtures("allow_fall_back_on_numpy")
class TestCorrelateInvalid(unittest.TestCase):
    @testing.with_requires("numpy>=1.18")
    @testing.for_all_dtypes()
    def test_correlate_empty(self, dtype):
        for xp in (numpy, cupy):
            a = xp.zeros((0,), dtype=dtype)
            with pytest.raises(ValueError):
                xp.correlate(a, a, mode=self.mode)

    @testing.for_all_dtypes()
    def test_correlate_ndim(self, dtype):
        for xp in (numpy, cupy):
            a = testing.shaped_arange((5, 10, 2), xp, dtype)
            b = testing.shaped_arange((3, 4, 4), xp, dtype)
            with pytest.raises(ValueError):
                xp.correlate(a, b, mode=self.mode)

    @testing.for_all_dtypes()
    def test_correlate_zero_dim(self, dtype):
        for xp in (numpy, cupy):
            a = testing.shaped_arange((), xp, dtype)
            b = testing.shaped_arange((1,), xp, dtype)
            with pytest.raises(ValueError):
                xp.correlate(a, b, mode=self.mode)
