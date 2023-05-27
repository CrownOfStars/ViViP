import numpy as np
from thop import profile

def get_parameter_num(model):
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
    return ('Trainable Parameters: %.3fM' % parameters)

def get_macs(model,inputs):
    macs, _ = profile(model, (inputs,))
    return ("input shape:{} MACS: {}G".format(inputs.shape,macs/1e9))