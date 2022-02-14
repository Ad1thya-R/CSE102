#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  9 14:11:39 2022

@author: adithya.ravichandran
"""
import math


class Fraction:
    def __init__(self, numerator=1, denominator=1):
        x = numerator
        y = denominator
        if denominator != 0:
            self.__numerator = x // math.gcd(x, y)
            self.__denominator = y // math.gcd(x, y)
        else:
            self.__numerator = 1
            self.__denominator = 1

    def reduce(self):
        x = self.__numerator
        y = self.__denominator
        self.__numerator = x // math.gcd(x, y)
        self.__denominator = y // math.gcd(x, y)

    @property  # READ access to numerator
    def numerator(self):
        return self.__numerator

    @property  # read access to denominator
    def denominator(self):
        return self.__denominator

    def __str__(self):
        return "%d/%d" % (self.__numerator, self.__denominator)

    def __repr__(self):
        return "Fraction(%d, %d)" % (self.__numerator, self.__denominator)

    def __eq__(self, o):
        firstnum = self.__numerator * o.denominator
        secondnum = o.numerator * self.__denominator

        return firstnum == secondnum

    def __ne__(self, o):
        firstnum = self.__numerator * o.denominator
        secondnum = o.numerator * self.__denominator

        return firstnum != secondnum

    def __add__(self, o):

        self.__numerator = self.__numerator * o.denominator + self.__denominator * o.numerator
        self.__denominator = self.__denominator * o.denominator

        return self

    def __sub__(self, o):
        self.__numerator = self.__numerator * o.denominator - self.__denominator * o.numerator
        self.__denominator = self.__denominator * o.denominator

        return self

    def __neg__(self):
        self.__numerator = -self.__numerator
        return self

    def __mul__(self, o):
        self.__numerator = self.__numerator * o.numerator
        self.__denominator *= o.denominator
        return self


    def __truediv__(self, o):
        self.__numerator = self.__numerator * o.denominator
        self.__denominator = self.__denominator * o.numerator
        return self

