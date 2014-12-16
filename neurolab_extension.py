
import neurolab as nl

def net_init(net):
  print net.layers
  # for i, layer in enumerate(net.layers):
  #   print layer.np


def midpoint(layer):
    mid = layer.inp_minmax.mean(axis=1)
    for i, w in enumerate(layer.np['w']):
        layer.np['w'][i] = mid.copy()
    return
