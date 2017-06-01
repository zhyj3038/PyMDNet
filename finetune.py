import numpy as np

def finetune(sess, model, pos_data, neg_data, config):
  print('finetuining...')
  
  ## parameters
  n_pos = pos_data.shape[0]
  n_neg = neg_data.shape[0]
  train_pos_cnt = 0
  train_neg_cnt = 0
  neg_batch_size = config.batch_size_hnm*config.batch_acc_hnm
  pos_batch_size = config.batch_pos

  ## extract postive batches
  train_pos = np.array([]).astype(np.int)
  remain = pos_batch_size*config.maxiter
  while remain > 0:
    if train_pos_cnt == 0:
      train_pos_list = np.random.permutation(n_pos)

    train_pos = np.r_[train_pos, train_pos_list[train_pos_cnt:min(remain+train_pos_cnt,n_pos)]]
    train_pos_cnt = min(remain+train_pos_cnt,n_pos) % n_pos
    remain = pos_batch_size*config.maxiter - train_pos.shape[0]

  ## extract negative batches
  train_neg = np.array([]).astype(np.int)
  remain = neg_batch_size*config.maxiter
  while remain > 0:
    if train_neg_cnt == 0:
      train_neg_list = np.random.permutation(n_neg)

    train_neg = np.r_[train_neg, train_neg_list[train_neg_cnt:min(remain+train_neg_cnt,n_neg)]]
    train_neg_cnt = min(remain+train_neg_cnt,n_neg) % n_neg
    remain = neg_batch_size*config.maxiter - train_neg.shape[0]

  ## finetune
  sess.run(model.lr_rate.assign(config.lr_rate))
  for t in range(config.maxiter):

    # hard negative mining
    scores = np.array([])
    for h in range(config.batch_acc_hnm):
      neg_start = neg_batch_size * t + config.batch_size_hnm * h
      neg_batch = neg_data[train_neg[neg_start:(neg_start+config.batch_size_hnm)]]

      # calculate score
      score = sess.run(model.layers['fc6'], feed_dict={model.layers['input']:neg_batch})[:, 0, 0, 0]
      scores = np.r_[scores, score]

    # find the maximum batch_neg boxes
    neg_data_ind = np.argsort(scores)[:-(config.batch_neg+1):-1]
    patch_neg = neg_data[train_neg[neg_batch_size * t + neg_data_ind]]

    # find the maximum batch_pos boxes
    patch_pos = pos_data[train_pos[pos_batch_size*t:pos_batch_size*(t+1)]]

    # pack positive batch and negative batch
    boxes = np.r_[patch_pos, patch_neg]
    gts = np.repeat(np.identity(2), [config.batch_pos, config.batch_neg], axis = 0)

    # shuffle
    inds = np.random.permutation(config.batch_size)
    boxes = boxes[inds]
    gts = gts[inds]

    sess.run(model.trainable[-1], feed_dict={model.layers['input']: boxes, model.layers['y']:gts})