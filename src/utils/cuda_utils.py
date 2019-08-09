#!/usr/bin/env python

# https://gist.github.com/f0k/63a664160d016a491b2cbea15913d549


import ctypes

from src.utils.logger import HuliLogging

logger = HuliLogging.get_logger(__name__)


# Some constants taken from cuda.h
CUDA_SUCCESS = 0
CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT = 16
CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR = 39
CU_DEVICE_ATTRIBUTE_CLOCK_RATE = 13
CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE = 36


def _ConvertSMVer2Cores(major, minor):
    # Returns the number of CUDA cores per multiprocessor for a given
    # Compute Capability version. There is no way to retrieve that via
    # the API, so it needs to be hard-coded.
    return {(1, 0): 8,
            (1, 1): 8,
            (1, 2): 8,
            (1, 3): 8,
            (2, 0): 32,
            (2, 1): 48,
            }.get((major, minor), 192)  # 3.0 and above


def gpu_info(print_prefix=None, print_fn=logger.debug, print_err_fn=logger.error):
    _print_fn, _print_err_fn = print_fn, print_err_fn
    if print_prefix is not None:
        assert isinstance(print_prefix, str)
        _print_fn = lambda *argv, **kwargs: print_fn(('%s ' + argv[0]) % print_prefix, *argv[1:], **kwargs)
        _print_err_fn = lambda *argv, **kwargs: print_err_fn(('%s ' + argv[0]) % print_prefix, *argv[1:], **kwargs)


    libnames = ('libcuda.so', 'libcuda.dylib', 'cuda.dll')
    for libname in libnames:
        try:
            cuda = ctypes.CDLL(libname)
        except OSError:
            continue
        else:
            break
    else:
        raise OSError("could not load any of: " + ' '.join(libnames))

    nGpus = ctypes.c_int()
    name = b' ' * 100
    cc_major = ctypes.c_int()
    cc_minor = ctypes.c_int()
    cores = ctypes.c_int()
    threads_per_core = ctypes.c_int()
    clockrate = ctypes.c_int()
    freeMem = ctypes.c_size_t()
    totalMem = ctypes.c_size_t()

    result = ctypes.c_int()
    device = ctypes.c_int()
    context = ctypes.c_void_p()
    error_str = ctypes.c_char_p()

    result = cuda.cuInit(0)
    if result != CUDA_SUCCESS:
        cuda.cuGetErrorString(result, ctypes.byref(error_str))
        _print_err_fn("cuInit failed with error code %d: %s" % (result, error_str.value.decode()))
        return 1
    result = cuda.cuDeviceGetCount(ctypes.byref(nGpus))
    if result != CUDA_SUCCESS:
        cuda.cuGetErrorString(result, ctypes.byref(error_str))
        _print_err_fn("cuDeviceGetCount failed with error code %d: %s" % (result, error_str.value.decode()))
        return 1

    _print_fn("Found %d CUDA device%s." % (nGpus.value, '' if nGpus.value == 1 else 's'))
    for i in range(nGpus.value):
        prefix = '[GPU:%d]' % i
        result = cuda.cuDeviceGet(ctypes.byref(device), i)
        if result != CUDA_SUCCESS:
            cuda.cuGetErrorString(result, ctypes.byref(error_str))
            _print_err_fn("%s cuDeviceGet failed with error code %d: %s" % (prefix, result, error_str.value.decode()))
            return 1

        # Name
        if cuda.cuDeviceGetName(ctypes.c_char_p(name), len(name), device) == CUDA_SUCCESS:
            _print_fn("%s Name: %s" % (prefix, name.split(b'\0', 1)[0].decode()))
        # Compute Capability
        if cuda.cuDeviceComputeCapability(ctypes.byref(cc_major), ctypes.byref(cc_minor), device) == CUDA_SUCCESS:
            _print_fn("%s Compute Capability: %d.%d" % (prefix, cc_major.value, cc_minor.value))
        # Processors / Cores / Threads
        if cuda.cuDeviceGetAttribute(ctypes.byref(cores), CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
                                     device) == CUDA_SUCCESS:
            _print_fn("%s Multiprocessors: %d" % (prefix, cores.value))
            _print_fn("%s CUDA Cores: %d" % (prefix, cores.value * _ConvertSMVer2Cores(cc_major.value, cc_minor.value)))
            if cuda.cuDeviceGetAttribute(ctypes.byref(threads_per_core),
                                         CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR,
                                         device) == CUDA_SUCCESS:
                _print_fn("%s Concurrent threads: %d" % (prefix, cores.value * threads_per_core.value))
        # GPU Clock
        if cuda.cuDeviceGetAttribute(ctypes.byref(clockrate), CU_DEVICE_ATTRIBUTE_CLOCK_RATE,
                                     device) == CUDA_SUCCESS:
            _print_fn("%s GPU clock: %g MHz" % (prefix, clockrate.value / 1000.))
        # Memory clock
        if cuda.cuDeviceGetAttribute(ctypes.byref(clockrate), CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE,
                                     device) == CUDA_SUCCESS:
            _print_fn("%s Memory clock: %g MHz" % (prefix, clockrate.value / 1000.))
        # Memory
        result = cuda.cuCtxCreate(ctypes.byref(context), 0, device)
        if result != CUDA_SUCCESS:
            cuda.cuGetErrorString(result, ctypes.byref(error_str))
            _print_err_fn("%s cuCtxCreate failed with error code %d: %s" % (prefix, result, error_str.value.decode()))
        else:
            result = cuda.cuMemGetInfo(ctypes.byref(freeMem), ctypes.byref(totalMem))
            if result == CUDA_SUCCESS:
                _print_fn("%s Total Memory: %ld MiB" % (prefix, totalMem.value / 1024 ** 2))
                _print_fn("%s Free Memory: %ld MiB" % (prefix, freeMem.value / 1024 ** 2))
            else:
                cuda.cuGetErrorString(result, ctypes.byref(error_str))
                _print_err_fn("%s cuMemGetInfo failed with error code %d: %s"
                             % (prefix, result, error_str.value.decode()))
            cuda.cuCtxDetach(context)
    return 0