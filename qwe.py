import dpnp
import numpy

a = numpy.array([[1,2],[4,5]],dtype='f8')
a_dp = dpnp.array(a)
mode = "reduced"
res = numpy.linalg.qr(a, mode=mode)
res_dp = dpnp.linalg.qr(a_dp, mode=mode)

print(res)

print(res_dp)
