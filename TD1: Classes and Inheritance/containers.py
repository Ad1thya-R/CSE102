#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  9 14:42:38 2022

@author: adithya.ravichandran
"""


class Rdict:
    def __init__(self, fwd, bwd):
        self.__fwd = {}
        self.__bwd = {}

    def associate(self, a, b):
        if a in self.__bwd.keys() or b in self.__fwd.keys():
            raise KeyError()
        else:
            self.__fwd[a] = b
            self.__bwd[b] = a

    @property  # READ access to numerator
    def fwd(self):
        return self.__fwd

    @property  # read access to denominator
    def bwd(self):
        return self.__bwd

    def __len__(self):
        return len(self.__fwd)

    def __getitem__(self, key):
        n, thing = key
        if n > 0:
            return self.__fwd[thing]
        else:
            return self.__bwd[thing]

    def __setitem__(self, key, value):
        n, k = key
        if n > 0:
            self.associate(k, value)
        else:
            self.associate(value, k)
