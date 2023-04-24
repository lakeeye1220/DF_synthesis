#*
# @file Different utility functions
# Copyright (c) Zhewei Yao, Amir Gholami
# All rights reserved.
# This file is part of PyHessian library.
#
# PyHessian is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# PyHessian is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with PyHessian.  If not, see <http://www.gnu.org/licenses/>.
#*

import torch
import math
from torch.autograd import Variable
import numpy as np


def group_product(xs, ys):
    """
    the inner product of two lists of variables xs,ys
    :param xs:
    :param ys:
    :return:
    """
    return sum([torch.sum(x * y) for (x, y) in zip(xs, ys)])


def group_add(params, update, alpha=1):
    """
    params = params + update*alpha
    :param params: list of variable
    :param update: list of data
    :return:
    """
    for i, p in enumerate(params):
        params[i].data.add_(update[i] * alpha)
    return params


def normalization(v):
    """
    normalization of a list of vectors
    return: normalized vectors v
    """
    s = group_product(v, v)
    s = s**0.5
    s = s.cpu().item()
    v = [vi / (s + 1e-6) for vi in v]
    return v


#def get_params_grad(model):
def get_params_grad(data):
    """
    get model parameters and corresponding gradients
    """
    params = []
    grads = []
    '''
    for param in model.parameters():
        if not param.requires_grad:
            continue
        params.append(param)
        grads.append(0. if param.grad is None else param.grad + 0.)
    '''
    #for d in data:
    #print(d)
    #for d in data:
    inputs, targets = data
    print("inputs shape :",inputs.shape)
    inputs.requires_grad_(True)
    #print("input's gradient: ",inputs.grad)
    params.append(inputs)
    grads.append(0. if inputs.grad is None else inputs.grad + 0.)
    print("length of inputs : ",len(params))
    print("parameter shape : ",params[0].shape)
    print("length of grads : ",len(grads))
    #print("gradines: ",grads)
    #print("gradient shape : ",grads[0].shape)
    #img.requires_grad_(True)
    #params.append(img)
    #grads.append(0. if img.grad is None else img.grad +0.)s
    return params, grads

def get_params_grad_input(imgs):
    """
    get model parameters and corresponding gradients
    """
    params = []
    grads = []
    '''
    for param in model.parameters():
        if not param.requires_grad:
            continue
        params.append(param)
        grads.append(0. if param.grad is None else param.grad + 0.)
    '''
    for inputs in imgs:
        inputs.requires_grad_(True)
        params.append(inputs)
        grads.append(0. if inputs.grad is None else inputs.grad +0.)
    return params, grads



def hessian_vector_product(gradsH, params, v):
    """
    compute the hessian vector product of Hv, where
    gradsH is the gradient at the current point,
    params is the corresponding variables,
    v is the vector.
    """
    '''
    print("gradsH shape :",len(gradsH))
    print("gradsH[0] shape :",gradsH[0])
    print("gradsH[1] shape :",gradsH[1])
    print("param shape :",len(params))
    print("param[0] shape :",params[0].shape)
    print("param[1] shape :",params[1].shape)
    '''
    hv = torch.autograd.grad(gradsH,
                             params,
                             grad_outputs=v,
                             only_inputs=True,
                             retain_graph=True)
    return hv


def orthnormal(w, v_list):
    """
    make vector w orthogonal to each vector in v_list.
    afterwards, normalize the output w
    """
    for v in v_list:
        w = group_add(w, v, alpha=-group_product(w, v))
    return normalization(w)
