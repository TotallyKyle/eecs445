
import neurolab as nl

def net_init(net, init_values):
  for i, layer in enumerate(net.layers):
    layer.np = init_values[i]

def midpoint(layer):
    mid = layer.inp_minmax.mean(axis=1)
    for i, w in enumerate(layer.np['w']):
        layer.np['w'][i] = mid.copy()
    return
