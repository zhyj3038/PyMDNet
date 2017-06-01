import tensorflow as tf
import numpy as np
from models import MDNet
import reader
import proc
import os
import utils
import finetune
from proc import overlap_ratio
import time
from bbox_regressor import train_bbox_regressor, predict_bbox_regressor
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib
from PIL import Image

class Config(object):
  def __init__(self, im_size):
    # image size
    self.im_size = im_size
    
    # bbox regression
    self.bbreg = False
    self.bbreg_n_samples = 1024

    # learning policy
    self.batch_size = 128
    self.batch_pos = 32
    self.batch_neg = 96
    self.momentum = 0.9
    self.weight_decay = 0.0005
    self.lr_rate = 0.0001
  
    # initial training policy
    self.lr_rate_init = 0.0001
    self.maxiter_init = 30
    self.n_pos_init = 500
    self.n_neg_init = 5000
    self.pos_thr_init = 0.7
    self.neg_thr_init = 0.5

    # update policy
    self.lr_rate_update = 0.0003
    self.maxiter_update = 10
    self.n_pos_update = 50
    self.n_neg_update = 200
    self.pos_thr_update = 0.7
    self.neg_thr_update = 0.3

    # interval for long-term update
    self.update_interval = 10
    self.update_thr = 0.5

    # data gathering policy
    self.n_frames_long = 100
    self.n_frames_short = 20
  
    # cropping policy
    self.input_size = 107
    self.crop_mode = 'wrap'
    self.crop_padding = 16

    # scaling policy
    self.scale_factor = 1.05

    # sampling policy
    self.n_samples = 256
    self.trans_f = 0.6
    self.scale_f = 1
    self.lr_rates = {'conv': 1.0, 'bias': 2.0, 'fc6-conv': 10.0, 'fc6-bias': 20.0}
    self.weight_decay = 0.0005
    self.momentum = 0.9

    # finetune parameter
    self.batch_size_hnm = self.batch_size
    self.batch_acc_hnm = 8

def mdnet_run(sess, model, region, images, config, display):
  targetLoc = region
  n_frames = len(images)
  img = proc.load_image(images[0])
  tres = open('tres.txt', 'w')
  tres.close()
  plt.ion()
  
  ######################### bbox regressor #########################
  if config.bbreg:
    pos_examples = utils.gen_samples('uniform_aspect', targetLoc, config.bbreg_n_samples*10, 0.3, 10, config)
    r = overlap_ratio(pos_examples, targetLoc)
    pos_examples = pos_examples[r>0.6]
    pos_examples = pos_examples[np.random.choice(pos_examples.shape[0], min(pos_examples.shape[0], config.bbreg_n_samples))]
    if pos_examples.shape[0] < config.bbreg_n_samples:
      pos_examples = pos_examples[:pos_examples[0] // config.batch_size * config.batch_size]

    # evaluate candidates
    feats = np.array([])
    for i in range(pos_examples.shape[0] // config.batch_size):
      sample = pos_examples[config.batch_size * i: config.batch_size * (i+1)]
      sample_im = proc.load_box(img, sample, img_input=True)
      feat = sess.run(model.layers['conv3'], feed_dict={model.layers['input']:sample_im})
      feat = feat.reshape(config.batch_size, -1)
      if feats.size == 0:
        feats = feat
      else:
        feats = np.r_[feats, feat]
    print('bbox regression features extracted')
    bbox_reg = train_bbox_regressor(feats, pos_examples, targetLoc)

  ################## finetune on the first frame ###################
  print('Finetune on the first frame...')
  # generate positive examples
  pos_examples = utils.gen_samples('gaussian', targetLoc, config.n_pos_init*2, 0.1, 0.5, config)
  r = overlap_ratio(pos_examples, targetLoc)
  pos_examples = pos_examples[r>config.pos_thr_init]
  pos_examples = pos_examples[np.random.choice(pos_examples.shape[0], min(pos_examples.shape[0], config.n_pos_init))]

  neg_examples = np.r_[utils.gen_samples('uniform', targetLoc, config.n_neg_init, 1, 10, config), \
                       utils.gen_samples('whole', targetLoc, config.n_neg_init, 0, 0, config)]
  r = overlap_ratio(neg_examples, targetLoc)
  neg_examples = neg_examples[r<config.neg_thr_init]
  neg_examples = neg_examples[np.random.choice(neg_examples.shape[0], min(neg_examples.shape[0], config.n_neg_init))]

  # prepare patches
  pos_data = proc.load_box(img, pos_examples, img_input=True)
  neg_data = proc.load_box(img, neg_examples, img_input=True)
  config.maxiter = config.maxiter_init
  config.lr_rate = config.lr_rate_init
  finetune.finetune(sess, model, pos_data, neg_data, config)

  ############# Prepare training data for online update ##############
  print('Preparing online updating data...')
  
  neg_examples = utils.gen_samples('uniform', targetLoc, config.n_neg_update*2, 2, 5, config)
  r = overlap_ratio(neg_examples, targetLoc)
  neg_examples = neg_examples[r<config.neg_thr_init]
  neg_examples = neg_examples[np.random.choice(neg_examples.shape[0], min(neg_examples.shape[0], config.n_neg_update))]

  total_pos_data = []
  total_neg_data = []
  total_pos_data.append(proc.load_box(img, pos_examples, img_input=True))
  total_neg_data.append(proc.load_box(img, neg_examples, img_input=True))

  ############################ tracking ##############################
  success_frames = np.array([0]).astype(np.int)
  result = np.array([targetLoc]).reshape(-1, 4)
  trans_f = config.trans_f
  scale_f = config.scale_f
  for To in range(1,len(images)):
    print(targetLoc)
    t = time.time()
    print('Processing frame %d/%d...'%(To+1,n_frames))
    
    img = proc.load_image(images[To])
    
    ## estimation
    # draw target candidates
    samples = utils.gen_samples('gaussian', targetLoc, config.n_samples, trans_f, scale_f, config)

    # evaluate candidates
    remain = config.n_samples
    scores = np.array([])
    feats = np.array([])
    while(remain>0):
      sample = samples[scores.shape[0]:scores.shape[0]+config.batch_size]
      sample_im = proc.load_box(img, sample, img_input=True)
      score, feat = sess.run([model.layers['fc6'], model.layers['conv3']], 
                       feed_dict={model.layers['input']:sample_im})
      
      score = score[:, 0, 0, 0]
      scores = np.r_[scores, score]
      
      feat = feat.reshape(config.batch_size, -1)
      if feats.size == 0:
        feats = feat
      else:
        feats = np.r_[feats, feat]
      
      remain = config.n_samples - scores.shape[0]
    
    # sort the bboxes
    inds = np.argsort(scores)[::-1]
    
    # generate prediction
    target_score = np.mean(scores[inds[:5]])
    targetLoc = np.round(np.mean(samples[inds[:5]], axis=0))
    result = np.r_[result, targetLoc.reshape(1, -1)]

    # extend search space in case of failure
    if target_score < 0:
      trans_f = min(1.5, 1.1*trans_f)
    else:
      trans_f = config.trans_f

    # bbox regression
    if config.bbreg and target_score > 0:
      X_ = feats[inds[:5]]
      bbox_ = samples[inds[:5]]
      pred_bboxes = predict_bbox_regressor(bbox_reg.model, X_, bbox_)
      targetLoc = np.round(np.mean(pred_bboxes, axis=0))
      result[-1] = targetLoc.reshape(1, -1)

    print(targetLoc)

    if display:
      im = Image.open(images[To])
      print(images[To])
      fig, ax = plt.subplots(1)
      ax.imshow(im)
      for i in range(10):
        rect = patches.Rectangle(samples[i, :2], samples[i, 2], samples[i, 3],linewidth=0.5,edgecolor='b',facecolor='none')
        ax.add_patch(rect)
      rect = patches.Rectangle(targetLoc[:2], targetLoc[2], targetLoc[3],linewidth=1,edgecolor='r',facecolor='none')
      ax.add_patch(rect)
      plt.imshow(im)
      fig.savefig(os.path.join('res', os.path.basename(images[To])))
      plt.close(fig)

    # prepare training data
    print(target_score)
    if target_score > config.update_thr:
      pos_examples = utils.gen_samples('gaussian', targetLoc, config.n_pos_init*2, 0.1, 0.5, config)
      r = overlap_ratio(pos_examples, targetLoc)
      pos_examples = pos_examples[r>config.pos_thr_update]
      pos_examples = pos_examples[np.random.choice(pos_examples.shape[0], min(pos_examples.shape[0], config.n_pos_update))]

      neg_examples = utils.gen_samples('uniform', targetLoc, config.n_neg_update*2, 2, 5, config)
      r = overlap_ratio(neg_examples, targetLoc)
      neg_examples = neg_examples[r<config.neg_thr_update]
      neg_examples = neg_examples[np.random.choice(neg_examples.shape[0], min(neg_examples.shape[0], config.n_neg_update))]

      total_pos_data.append(proc.load_box(img, pos_examples, img_input=True))
      total_neg_data.append(proc.load_box(img, neg_examples, img_input=True))

      success_frames = np.r_[success_frames, To]
      if success_frames.shape[0] > config.n_frames_long:
        tmp = total_pos_data[success_frames[-config.n_frames_long-1]]
        total_pos_data[success_frames[-config.n_frames_long-1]] = np.array([])
        del tmp
      #if success_frames.shape[0] > config.n_frames_short:
      #  tmp = total_neg_data[success_frames[-config.n_frames_short-1]]
      #  total_neg_data[success_frames[-config.n_frames_short-1]] = np.array([])
      #  del tmp

    else:
      total_pos_data.append(np.array([]).reshape(-1,4))
      total_neg_data.append(np.array([]).reshape(-1,4))

    # network update
    if ((To+1) % config.update_interval == 0 or target_score <= config.update_thr) and To != n_frames-1:
      print('##################### finetuning #######################')
      if target_score < config.update_thr: # short-term update
        pos_inds = success_frames[max(0,success_frames.shape[0]-config.n_frames_short):]
      else: # long-term update
        pos_inds = success_frames[max(0,success_frames.shape[0]-config.n_frames_long):]
      neg_inds = success_frames[max(0,success_frames.shape[0]-config.n_frames_short):]

      pos_data = np.concatenate([total_pos_data[pos_ind] for pos_ind in pos_inds])
      neg_data = np.concatenate([total_neg_data[neg_ind] for neg_ind in neg_inds])

      config.maxiter = config.maxiter_update
      config.lr_rate = config.lr_rate_update
      finetune.finetune(sess, model, pos_data, neg_data, config)

    elapsed = time.time() - t
    print("Elapsed Time: ", elapsed)
    with open('tres.txt', 'a') as tres:
      tres.write("Elapsed Time: "+str(elapsed))
      tres.write('\n')

  with open('res.txt', 'w') as f:
    for i in range(result.shape[0]):
      f.write(','.join([str(num) for num in result[i]]))
      f.write('\n')


def tracking(dataset, seq, display, restore_path):
  train_data = reader.read_seq(dataset, seq)
  im_size = proc.load_image(train_data.data[seq].frames[0]).shape[:2]
  config = Config(im_size)

  # create session and saver
  gpu_config = tf.ConfigProto(allow_soft_placement=True)
  sess = tf.InteractiveSession(config=gpu_config)

  # load model, weights
  model = MDNet(config)
  model.build_generator(config.batch_size, reuse=False, dropout=True)
  tf.global_variables_initializer().run()

  # create saver
  saver = tf.train.Saver([v for v in tf.global_variables() if ('conv' in v.name or 'fc4' in v.name or 'fc5' in v.name) \
                          and 'lr_rate' not in v.name], max_to_keep=50)
  
  # restore from model
  saver.restore(sess, restore_path)
  
  # run mdnet
  mdnet_run(sess, model, train_data.data[seq].gts[0], train_data.data[seq].frames, config, display)


def get_params():
  parser = argparse.ArgumentParser()
  parser.add_argument('--no_display', action='store_true', help='disable display')
  parser.add_argument('--dataset', choices=['otb', 'vot2013', 'vot2014', 'vot2015'], help='choose pretrained dataset: [otb/vot2013/vot2014/vot2015]')
  parser.add_argument('--seq', default=None, help='specify the sequence name')
  parser.add_argument('--load_path', default=None, help='initial model path')
  return parser.parse_args()

if __name__ == '__main__':
  params = get_params()
  tracking(params.dataset, params.seq, display=(not params.no_display), restore_path=params.load_path)
