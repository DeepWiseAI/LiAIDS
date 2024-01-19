# -*- coding:utf-8 -*-

"""
    @File : __init__.py
    @Time : 2020/10/27 10:32
    @Author : sxwang
"""
from __future__ import absolute_import
from .TripleSplitNetHBP_220428 import TripleSplitNetHBP_220428


__factory = {
    'TripleSplitNetHBP_220428':TripleSplitNetHBP_220428,
}


def get_names():
    return __factory.keys()


def init_model(name, *args, **kwargs):
    if name not in __factory.keys():
        raise KeyError("Unknown models: {}".format(name))
    return __factory[name](*args, **kwargs)
