from utils import Data
from PIL import Image
import numpy as np
import tensorflow as tf
from skimage import io, transform

###########################################################################
#                            load_patch                                   #
###########################################################################
def load_box(inp, bboxes, img_input=False, norm=False):
  n = bboxes.shape[0]
  if img_input:
    im = inp
  else:
    im = load_image(inp, norm)

  res = np.zeros([n, 117, 117, 3])
  for i in range(n):
    img_crop = im_crop(im, bboxes[i])
    img_resize = transform.resize(img_crop, [117, 117])
    res[i] = img_resize
  return res

###########################################################################
#                            load_patch                                   #
###########################################################################
def load_patch(paths, bboxes, norm=False):
  n = len(paths)

  res = np.zeros([n, 117, 117, 3])
  for i in range(n):
    path = paths[i]
    bbox = bboxes[i]
    img_crop = im_crop(load_image(path, norm=norm), bbox)
    img_resize = transform.resize(img_crop, [117, 117])
    res[i] = img_resize
  return res

###########################################################################
#                            load_image                                   #
###########################################################################
def load_image(path, norm=False):
  # load image
  img = io.imread(path)
  if len(img.shape) == 2:
    img = np.repeat(img[:, :, np.newaxis], 3, axis=2)
  if norm:
    img = img.astype(np.float32) / 255.0 - 0.5
  return img

###########################################################################
#                            im_crop                                      #
###########################################################################
def im_crop(im, bbox):
  bbox = np.around(bbox).astype(np.int)
  h, w, c = im.shape

  h_bot = np.maximum(0, bbox[1])
  h_top = np.minimum(h, bbox[1]+bbox[3])
  w_bot = np.maximum(0, bbox[0])
  w_top = np.minimum(w, bbox[0]+bbox[2])
  im_cropped = im[h_bot:h_top, w_bot:w_top]
  return im_cropped

###########################################################################
#                          seq2roidb                                      #
###########################################################################
def seq2roidb(seq_data,config):
  gts = seq_data.gts
  frames = seq_data.frames
  im = io.imread(frames[0])
  im_size = im.shape[:2]
  return sample_rois(frames, gts, im_size, config)

def sample_rois(frames, gts, im_size, config):
  '''
   rois = [ <img_path: string, pos_boxes: np.array, neg_boxes: np.array> ]
  '''
  rois = []
  
  for i in range(len(frames)):
    #print(str(i))
    target = gts[i]
    
    # sample postive boxes
    verbose = False
    pos_examples = np.array([]).reshape([0, 4])
    while len(pos_examples) < config.posPerFrame-1 :
      pos = sample(target, config.posPerFrame*5, im_size, config.scale_factor, 0.1, 5, False, verbose)
      r = overlap_ratio(pos,target)
      pos = pos[np.logical_and(r>config.pos_range[0],r<=config.pos_range[1])]

      if verbose:
        exit(0)

      if pos.shape[0] == 0:
        verbose = True
        continue
      index = np.arange(pos.shape[0])
      np.random.shuffle(index)
      index = index[:min(pos.shape[0],config.posPerFrame-pos_examples.shape[0]-1)]
      pos_examples = np.r_[pos_examples,pos[index]]  

    # sample negative boxes
    verbose = False
    neg_examples = np.array([]).reshape([0, 4])
    while len(neg_examples) < config.negPerFrame :
      neg = sample(target, config.negPerFrame*2, im_size, config.scale_factor, 2, 10, True, verbose)
      r = overlap_ratio(neg,target)
      neg = neg[np.logical_and(r>=config.neg_range[0],r<config.neg_range[1])]

      if verbose:
        exit(0)

      if neg.shape[0] == 0:
        verbose = True
        continue
      index = np.arange(neg.shape[0])
      np.random.shuffle(index)
      index = index[:min(neg.shape[0],config.negPerFrame-neg_examples.shape[0])]
      neg_examples = np.r_[neg_examples,neg[index]]   

    # pack into rois
    rois.append(Data())
    rois[-1].img_path = frames[i]
    rois[-1].pos_boxes = np.r_[pos_examples,target.reshape(1,-1)]
    rois[-1].neg_boxes = neg_examples
  return rois

def sample(gt, n, im_size, scale_factor, transfer_range, scale_range, valid, verbose=False):
  samp = np.array([gt[0]+gt[2]/2.0, gt[1]+gt[3]/2.0, gt[2], gt[3]])
  samples = np.repeat(np.reshape(samp, [1, -1]), n, axis=0)
  h, w = im_size
    
  if verbose:
    print(w, h)
    print(gt)
    print(samp)
    print(transfer_range)
    print(scale_range)

  samples[:, 0] = np.add(samples[:, 0], transfer_range*samp[2]*(np.random.rand(n)*2-1))
  samples[:, 1] = np.add(samples[:, 1], transfer_range*samp[3]*(np.random.rand(n)*2-1))
  samples[:, 2:]  = np.multiply(samples[:, 2:], np.power(scale_factor, scale_range*np.repeat(np.random.rand(n,1)*2-1,2,axis=1)))
  samples[:, 2] = np.maximum(0, np.minimum(w-5, samples[:,2]))
  samples[:, 3] = np.maximum(0, np.minimum(h-5, samples[:,3]))
  
  if verbose:
    print(samples[0])

  samples = np.c_[samples[:,0]-samples[:,2]/2, samples[:,1]-samples[:,3]/2, samples[:,2], samples[:,3]]
  
  if verbose:
    print(samples[0])

  if valid:
    samples[:,0] = np.maximum(0,np.minimum(w-samples[:,2],samples[:,0]))
    samples[:,1] = np.maximum(0,np.minimum(h-samples[:,3],samples[:,1]))
  else:
    samples[:,0] = np.maximum(0-samples[:,2]/2,np.minimum(w-samples[:,2]/2,samples[:,0]))
    samples[:,1] = np.maximum(0-samples[:,3]/2,np.minimum(h-samples[:,3]/2,samples[:,1]))
  
  if verbose:
    print(samples[0])
  return samples

###########################################################################
#                          overlap_ratio                                  #
###########################################################################
def overlap_ratio(boxes1, boxes2):
  # find intersection bbox
  x_int_bot = np.maximum(boxes1[:, 0], boxes2[0])
  x_int_top = np.minimum(boxes1[:, 0] + boxes1[:, 2], boxes2[0] + boxes2[2])
  y_int_bot = np.maximum(boxes1[:, 1], boxes2[1])
  y_int_top = np.minimum(boxes1[:, 1] + boxes1[:, 3], boxes2[1] + boxes2[3])

  # find intersection area
  dx = x_int_top - x_int_bot
  dy = y_int_top - y_int_bot
  area_int = np.where(np.logical_and(dx>0, dy>0), dx * dy, np.zeros_like(dx))

  # find union
  area_union = boxes1[:,2] * boxes1[:,3] + boxes2[2] * boxes2[3] - area_int

  # find overlap ratio
  ratio = np.where(area_union > 0, area_int/area_union, np.zeros_like(area_int))
  return ratio


###########################################################################
#                          overlap_ratio of two bboxes                    #
###########################################################################
def overlap_ratio_pair(boxes1, boxes2):
  # find intersection bbox
  x_int_bot = np.maximum(boxes1[:, 0], boxes2[:, 0])
  x_int_top = np.minimum(boxes1[:, 0] + boxes1[:, 2], boxes2[:, 0] + boxes2[:, 2])
  y_int_bot = np.maximum(boxes1[:, 1], boxes2[:, 1])
  y_int_top = np.minimum(boxes1[:, 1] + boxes1[:, 3], boxes2[:, 1] + boxes2[:, 3])

  # find intersection area
  dx = x_int_top - x_int_bot
  dy = y_int_top - y_int_bot
  area_int = np.where(np.logical_and(dx>0, dy>0), dx * dy, np.zeros_like(dx))

  # find union
  area_union = boxes1[:,2] * boxes1[:,3] + boxes2[:, 2] * boxes2[:, 3] - area_int

  # find overlap ratio
  ratio = np.where(area_union > 0, area_int/area_union, np.zeros_like(area_int))
  return ratio
