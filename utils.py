import numpy as np

class Data(object):
  pass

def gen_samples(tp, bb, n, trans_f, scale_f, config):
  h, w = config.im_size
  samp = np.array([bb[0]+bb[2]/2.0, bb[1]+bb[3]/2.0, bb[2], bb[3]])
  samples = np.repeat(np.reshape(samp, [1, -1]), n, axis=0)

  if tp == 'gaussian':
    samples[:, :2] = samples[:, :2] + trans_f*np.round(np.mean(bb[2:]))*np.maximum(-1, np.minimum(1, 0.5*np.random.randn(n,2)))
    samples[:, 2:] = samples[:, 2:] * np.repeat(np.power(config.scale_factor,
                                                         scale_f*np.maximum(-1, np.minimum(1, 0.5*np.random.randn(n, 1)))), 2, axis=1)
  elif tp == 'uniform':
    samples[:, :2] = samples[:, :2] + trans_f*np.round(np.mean(bb[2:]))*(np.random.rand(n,2)*2-1)
    samples[:, 2:] = samples[:, 2:] * np.repeat(np.power(config.scale_factor, scale_f*(2*np.random.rand(n, 1)-1)), 2, axis=1)
  elif tp == 'uniform_aspect':
    samples[:, :2] = samples[:, :2] + trans_f*np.round(np.repeat(bb[2:].reshape(1, -1), n, axis=0))*(np.random.rand(n,2)*2-1)
    samples[:, 2:] = samples[:, 2:] * np.power(config.scale_factor, np.random.rand(n, 2)*4-2)
    samples[:, 2:] = samples[:, 2:] * np.repeat(np.power(config.scale_factor, scale_f*np.random.rand(n, 1)), 2, axis=1)
  elif tp == 'whole':
    rg = np.round(np.array([bb[2]/2, bb[3]/2, w-bb[2]/2, h-bb[3]/2]))
    stride = np.round(np.array([bb[2]/5, bb[3]/5]))
    dx, dy, ds = np.meshgrid(np.arange(rg[0], rg[2]+1, stride[0]), np.arange(rg[1], rg[3]+1, stride[1]), np.arange(-5, 6))
    windows = np.c_[dx.reshape(-1,1), dy.reshape(-1,1), \
                    bb[2]*np.power(config.scale_factor, ds.reshape(-1,1)), \
                    bb[3]*np.power(config.scale_factor, ds.reshape(-1,1))]

    samples = np.array([]).reshape(0,4)
    while samples.shape[0] < n:
      samples = np.r_[samples, windows[np.random.choice(windows.shape[0], min(windows.shape[0], n-samples.shape[0]))]]

  samples[:,2] = np.maximum(10,np.minimum(w-10,samples[:,2]))
  samples[:,3] = np.maximum(10,np.minimum(h-10,samples[:,3]))

  samples = np.c_[samples[:, 0]-samples[:,2]/2, samples[:,1]-samples[:,3]/2, samples[:,2], samples[:,3]]
  samples[:,0] = np.maximum(-samples[:,2]/2,np.minimum(w-samples[:,2]/2, samples[:,0]))
  samples[:,1] = np.maximum(-samples[:,3]/2,np.minimum(h-samples[:,3]/2, samples[:,1]))
  return samples
