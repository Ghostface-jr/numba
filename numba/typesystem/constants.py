# -*- coding: utf-8 -*-

"""
Default rules for the typing of constants.
"""

from __future__ import print_function, division, absolute_import

import math
import types
import ctypes

import numba.typesystem
from numba.typesystem import typesystem
from numba.support.ctypes_support import is_ctypes, from_ctypes_value
from numba.support import cffi_support
from numba import numbawrapper

import numpy as np

#------------------------------------------------------------------------
# Class -> Type
#------------------------------------------------------------------------

def get_typing_defaults(u):
    """
    Get a simple table mapping Python classes to types.

    :param u: The type universe
    """
    typing_defaults = {
        float: u.double,
        bool: u.bool,
        complex: u.complex128,
        str: u.pointer(u.char),
    }
    return typing_defaults

#------------------------------------------------------------------------
# Class -> pyval -> Type
#------------------------------------------------------------------------

def get_default_typing_rules(u, typeof, promote):
    """
    Get a table mapping Python classes to handlers (value -> type)

    :param u: The type universe
    """
    table = {}
    def register(*classes):
        def dec(func):
            for cls in classes:
                table[cls] = func
            return lambda u, value: func(value)
        return dec

    @register(int, long)
    def type_int(value):
        if abs(value) < 1:
            bits = 0
        else:
            bits = math.ceil(math.log(abs(value), 2))

        if bits < 32:
            return u.int
        elif bits < 64:
            return u.int64
        else:
            raise ValueError("Cannot represent %s as int32 or int64", value)

    @register(np.ndarray)
    def type_ndarray(value):
        if isinstance(value, np.ndarray):
            from numba.support import numpy_support
            dtype = numpy_support.map_dtype(value.dtype)
            return u.array(dtype, value.ndim)
                           #is_c_contig=value.flags['C_CONTIGUOUS'],
                           #is_f_contig=value.flags['F_CONTIGUOUS'])

    @register(tuple, list, dict)
    def type_container(value):
        assert isinstance(value, (tuple, list, dict))

        if isinstance(value, dict):
            key_type = type_container(value.keys(), promote, typeof)
            value_type = type_container(value.values(), promote, typeof)
            return u.dict(key_type, value_type, size=len(value))

        if isinstance(value, tuple):
            container_type = u.tuple
        else:
            container_type = u.list

        if len(value) < 30:
            # Figure out base type if the container is not too large
            base_type = reduce(promote, (typeof(child) for child in value))
        else:
            base_type = u.object

        return container_type(base_type, size=len(value))

    table[np.dtype] = lambda value: from_numpy_dtype(value)
    table[types.ModuleType] = lambda value: u.module(value)
    table[typesystem.Type] = lambda value: u.metatype(value)

    return table

#------------------------------------------------------------------------
# Constant matching ({ pyval -> bool : pyval -> Type })
#------------------------------------------------------------------------

# TODO: Make this a well-defined (easily overridable) matching table
# E.g. { "numpy" : { is_numpy : get_type } }

def is_dtype_constructor(value):
    return isinstance(value, type) and issubclass(value, np.generic)

def is_registered(value):
    from numba.type_inference import module_type_inference
    return module_type_inference.is_registered(value)

def from_ctypes(value, u):
    result = from_ctypes_value(value)
    if result.is_function:
        pointer = ctypes.cast(value, ctypes.c_void_p).value
        return u.constfuncptr(value, pointer, result)
    else:
        return result

def from_cffi(value, u):
    signature = cffi_support.get_signature(value)
    pointer = cffi_support.get_pointer(value)
    return u.constfuncptr(value, pointer, signature)

def from_typefunc(value, u):
    from numba.type_inference import module_type_inference
    result = module_type_inference.module_attribute_type(value)
    if result is not None:
        return result
    else:
        return u.KnownValueType(value)

is_numba_exttype = lambda value: hasattr(type(value), '__numba_ext_type')
is_NULL = lambda value: value is numba.NULL
is_autojit_func = lambda value: isinstance(
    value, numbawrapper.NumbaSpecializingWrapper)

def get_default_match_table(ts):
    u = ts.universe

    table = {
        is_dtype_constructor:
            lambda value: numba.typesystem.from_numpy_dtype(np.dtype(value)),
        is_ctypes:
            lambda value: from_ctypes(value, u),
        cffi_support.is_cffi_func:
            lambda value: from_cffi(value, u),
        is_numba_exttype:
            lambda value: getattr(type(value), '__numba_ext_type'),
        numbawrapper.is_numba_wrapper:
            lambda value: u.jitfunctype(value),
        is_autojit_func:
            lambda value: u.autojitfunctype(value),
        is_registered:
            lambda value: from_typefunc(value, u),
    }

    return table


def get_constant_typer(universe, typeof, promote):
    typetable = get_typing_defaults(universe)
    handler_table = get_default_typing_rules(universe, typeof, promote)
    return typesystem.ConstantTyper(universe, typetable, handler_table)