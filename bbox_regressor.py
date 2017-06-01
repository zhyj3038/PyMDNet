import numpy as np
from numpy import dot, diag, sqrt
from numpy.linalg import eig, inv, cholesky, lstsq
from proc import overlap_ratio
from utils import Data

def train_bbox_regressor(X, bbox, gt):
  config = Data()
  config.min_overlap = 0.6
  config.delta = 1000
  config.method = 'ridge_reg_chol'

  # get training groundtruth
  Y, O = get_examples(bbox, gt)
  X = X[O>config.min_overlap]
  Y = Y[O>config.min_overlap]

  # add bias
  X = np.c_[X, np.ones([X.shape[0], 1])]

  # center and decorrelate targets
  mu = np.mean(Y, axis=0).reshape(1, -1)
  Y = Y - mu
  S = dot(Y.T, Y) / Y.shape[0]
  D, V = eig(S)
  T = dot(dot(V, diag(1.0/sqrt(D+0.001))), V.T)
  T_inv = dot(dot(V, diag(sqrt(D+0.001))), V.T)
  Y = dot(Y, T)
  
  model = Data()
  model.mu = mu
  model.T = T
  model.T_inv = T_inv
  model.Beta = np.c_[solve(X, Y[:, 0], config.delta, config.method),
                     solve(X, Y[:, 1], config.delta, config.method),
                     solve(X, Y[:, 2], config.delta, config.method),
                     solve(X, Y[:, 3], config.delta, config.method)]

  # pack 
  bbox_reg = Data()
  bbox_reg.model = model
  bbox_reg.config = config
  return bbox_reg

def predict_bbox_regressor(model, feat, ex_boxes):
  if ex_boxes.size == 0:
    return np.array([]).reshape(-1, 4)

  # predict regression targets
  Y = np.dot(feat, model.Beta[:-1]) + model.Beta[-1]

  # invert transformation
  Y = dot(Y, model.T_inv)

  # read out prediction
  dst_size = Y[:, 2:]
  dst_ctr = Y[:, 2:]

  src_size = ex_boxes[:, 2:]
  src_ctr = ex_boxes[:, :2] + 0.5 * src_size

  pred_size = np.exp(dst_size) * src_size
  pred_ctr = dst_ctr * src_ctr + src_ctr

  pred = np.c_[pred_ctr - 0.5 * pred_size, pred_size]

  return pred

def get_examples(bbox, gt):
  # compute overlap ratio
  O = overlap_ratio(bbox, gt)
  
  # compute answer
  src_size = bbox[:, 2:]
  src_ctr = bbox[:, :2] + 0.5 * src_size

  gt_size = gt[2:]
  gt_ctr = gt[:2] + 0.5 * gt_size

  dst_size = np.log(gt_size / src_size)
  dst_ctr = (gt_ctr - src_ctr) * 1.0 / src_ctr

  Y = np.c_[dst_ctr, dst_size]

  return Y, O

def solve(A, y, delta, method):
  if method == 'ridge_reg_chol':
    R = cholesky(dot(A.T, A) + delta*np.identity(A.shape[1]))
    z = lstsq(R.T, dot(A.T, y))[0]
    x = lstsq(R, z)[0]
  elif method == 'ridge_reg_inv':
    x = dot(dot(inv(dot(A.T, A) + delta*np.identity(A.shape[1])), A.T), y)
  elif method == 'ls_mldivide':
    if delta > 0:
      print('ignoring lambda; no regularization used')
    x = lstsq(A, y)[0]
  loss = 0.5 * (dot(A, x) - y) **2
  return x.reshape(-1, 1)
  