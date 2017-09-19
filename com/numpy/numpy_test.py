#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/9/19 22:12
# @Author  : Jasontang
# @Site    :
# @File    : e.py
# @ToDo    :

from numpy import *

a = random.rand(4, 4)
print 'a=\n', a

randMat = mat(random.rand(4, 4))

print 'randMat=\n', randMat

invRandMat = randMat.I

print 'invRandMat=\n', invRandMat
print type(invRandMat)

myEye = randMat * invRandMat

print myEye - eye(4)
