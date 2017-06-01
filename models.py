import tensorflow as tf
import numpy as np

class MDNet(object):
  def __init__(self, config):
    self.layers  = {}
    self.weights = {}
    self.biases  = {}
    self.losses  = {}
    self.regular_losses = {}
    self.trainable = {}
    self.summaries = {}

    # set parameters
    self.lr_rates = {}
    for key, val in config.lr_rates.iteritems():
      self.lr_rates[key] = tf.get_variable('lr_rates/'+key, initializer=tf.constant(val), dtype=tf.float32)

    self.momentum = tf.get_variable('momentum', initializer=tf.constant(config.momentum), dtype=tf.float32)
    self.weight_decay = tf.get_variable('weight_decay', initializer=tf.constant(config.weight_decay), dtype=tf.float32)
    self.lr_rate = tf.get_variable('lr_rate', initializer=tf.constant(config.lr_rate), dtype=tf.float32)

  def build_trainer(self, K, batch_size=None, dropout=True, regularization=True):
    # create shared layers
    (self.feed('input', [None, 117, 117, 3])
        .conv('conv1', 7, 2, 96, 0.01, 0.1)
        .lrn('norm1', 2, 0.00010000000475, 0.75)
        .max_pool('pool1', 3, 2)
        .conv('conv2', 5, 2, 256, 0.01, 0.1)
        .lrn('norm2', 2, 0.00010000000475, 0.75)
        .max_pool('pool2', 3, 2)
        .conv('conv3', 3, 1, 512, 0.01, 0.1)
        .conv('fc4', 3, 1, 512, 0.01, 0.1))

    if dropout:
      self.dropout('drop4')
    
    self.conv('fc5', 1, 1, 512, 0.01, 0.1)
    
    if dropout:
      self.dropout('drop5')

    # domain-specific layers
    for i in range(K):
      istr = str(i)
      if dropout:
        self.conv('fc6-'+istr, 1, 1, 2, 0.01, 0, input=self.layers['drop5'], relu=False)
      else:
        self.conv('fc6-'+istr, 1, 1, 2, 0.01, 0, input=self.layers['fc5'], relu=False)
      self.feed('y-'+istr, [None, 2])
      self.softmax_cross_entropy('loss-'+istr, self.layers['fc6-'+istr], self.layers['y-'+istr], 2)

    # regularization loss
    if regularization:
      (self.create_regularization('conv1')
         .create_regularization('conv2')
         .create_regularization('conv3')
         .create_regularization('fc4')
         .create_regularization('fc5'))
      for i in range(K):
        self.create_regularization('fc6-'+str(i))  

    # prepare optimization tasks
    for i in range(K):
        print('building trainable '+str(i))
        self.optimize(i, regularization=regularization)

  def build_generator(self, batch_size=None, reuse=True, dropout=True, regularization=True):
    # create layers
    (self.feed('input', [None, 117, 117, 3])
        .conv('conv1', 7, 2, 96, 0.01, 0.1, reuse=reuse)
        .lrn('norm1', 2, 0.00010000000475, 0.75)
        .max_pool('pool1', 3, 2)
        .conv('conv2', 5, 2, 256, 0.01, 0.1, reuse=reuse)
        .lrn('norm2', 2, 0.00010000000475, 0.75)
        .max_pool('pool2', 3, 2)
        .conv('conv3', 3, 1, 512, 0.01, 0.1, reuse=reuse)
        .conv('fc4', 3, 1, 512, 0.01, 0.1, reuse=reuse))
    if dropout:
      self.dropout('drop4')
    
    self.conv('fc5', 1, 1, 512, 0.01, 0.1, reuse=reuse)
    
    if dropout:
      self.dropout('drop5')

    (self.conv('fc6', 1, 1, 2, 0.01, 0, relu=False)
         .feed('y', [None, 2])
         .softmax_cross_entropy('loss', self.layers['fc6'], self.layers['y'], 2))

    # regularization loss
    if regularization:
      (self.create_regularization('fc4')
         .create_regularization('fc5')
         .create_regularization('fc6'))

    # prepare optimization tasks
    self.optimize(reuse=reuse, regularization=regularization)

  def load(self, data_path, session, ignore_missing=False):
    '''Load network weights.
    data_path: The path to the numpy-serialized network weights
    session: The current TensorFlow session
    ignore_missing: If true, serialized weights for missing layers are ignored.
    '''
    data_dict = np.load(data_path).item()
    for op_name in data_dict:
      with tf.variable_scope(op_name, reuse=True):
        for param_name, data in data_dict[op_name].iteritems():
          try:
            print 'loading...', op_name, '_', param_name
            var = tf.get_variable(param_name)
            session.run(var.assign(data))
          except ValueError:
            if not ignore_missing:
              raise

  def optimize(self, branch=None, reuse=False, regularization=True):
    istr = str(branch)
    
    # prepare trainable variables
    if branch is None:
      conv_weights = [self.weights['fc4'],
                    self.weights['fc5']]
      conv_biases =  [self.biases['fc4'],
                    self.biases['fc5']]
    else:
      conv_weights = [self.weights['conv1'],
                    self.weights['conv2'],
                    self.weights['conv3'],
                    self.weights['fc4'],
                    self.weights['fc5']]
      conv_biases =  [self.biases['conv1'],
                    self.biases['conv2'],
                    self.biases['conv3'],
                    self.biases['fc4'],
                    self.biases['fc5']]
    
    if branch is None:
      last_weight = [self.weights['fc6']]
      last_bias   = [self.biases['fc6']]
      total_loss = self.losses['loss']
      if regularization:
        total_loss += self.regular_losses['fc4'] + \
                      self.regular_losses['fc5'] + \
                      self.regular_losses['fc6']
    else:
      last_weight = [self.weights['fc6-'+istr]]
      last_bias   = [self.biases['fc6-'+istr]]
      total_loss = self.losses['loss-'+istr]
      if regularization:
        total_loss += self.regular_losses['conv1'] + \
                      self.regular_losses['conv2'] + \
                      self.regular_losses['conv3'] + \
                      self.regular_losses['fc4'] + \
                      self.regular_losses['fc5'] + \
                      self.regular_losses['fc6-'+istr]

    # prepare gradients
    grads = tf.gradients(total_loss, conv_weights+conv_biases+last_weight+last_bias)
    grad1 = grads[:len(conv_weights)]
    grad2 = grads[len(conv_weights):len(conv_weights)+len(conv_biases)]
    grad3 = [grads[-2]]
    grad4 = [grads[-1]]

    # prepare layer-wise optimizer
    opt1 = tf.train.MomentumOptimizer(learning_rate=self.lr_rate*self.lr_rates['conv'], momentum=self.momentum)
    opt2 = tf.train.MomentumOptimizer(learning_rate=self.lr_rate*self.lr_rates['bias'], momentum=self.momentum)
    opt3 = tf.train.MomentumOptimizer(learning_rate=self.lr_rate*self.lr_rates['fc6-conv'], momentum=self.momentum)
    opt4 = tf.train.MomentumOptimizer(learning_rate=self.lr_rate*self.lr_rates['fc6-bias'], momentum=self.momentum)

    # prepare optimization
    train_op3 = opt3.apply_gradients(zip(grad3, last_weight))
    train_op4 = opt4.apply_gradients(zip(grad4, last_bias))

    if reuse:
      tf.get_variable_scope().reuse_variables()
    train_op1 = opt1.apply_gradients(zip(grad1, conv_weights))
    train_op2 = opt2.apply_gradients(zip(grad2, conv_biases))

    # group together
    if branch is None:
      self.trainable[-1] = tf.group(train_op1, train_op2, train_op3, train_op4)
    else:
      self.trainable[branch] = tf.group(train_op1, train_op2, train_op3, train_op4)

  def create_regularization(self, name):
    self.regular_losses[name] = 0.5 * self.weight_decay * tf.nn.l2_loss(self.weights[name])
    return self

  def feed(self, name, shape):
    self.layers[name] = tf.placeholder(dtype=tf.float32, shape=shape)
    self.last_layer = self.layers[name]
    return self
  
  def softmax_cross_entropy(self, name, logits, labels, num_classes):
    self.losses[name] = tf.nn.softmax_cross_entropy_with_logits( \
                          logits = tf.reshape(logits, [-1, num_classes]), \
                          labels  = tf.reshape(labels, [-1, num_classes]))
    return self

  def conv(self, name, filter_size, stride, num_output, stddev, bias, layer_name=None, input=None, reuse=False, relu=True):
    if input is None:
      input = self.last_layer
    with tf.variable_scope(name) as scope:
      if reuse:
        scope.reuse_variables()

      weight = tf.get_variable('weights', dtype=tf.float32, \
                  initializer=tf.random_normal([filter_size, filter_size, int(input.shape[3]), num_output], \
                  stddev=stddev))
      bias = tf.get_variable('biases', dtype=tf.float32, initializer=np.ones(num_output, dtype=np.float32)*bias)
      self.weights[name] = weight
      self.biases[name] = bias

    if layer_name is not None:
      name = layer_name

    if relu:
      self.layers[name] = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(input, weight, [1, stride, stride, 1], "VALID"), bias))
    else:
      self.layers[name] = tf.nn.bias_add(tf.nn.conv2d(input, weight, [1, stride, stride, 1], "VALID"), bias)
    self.last_layer = self.layers[name]
    return self

  def max_pool(self, name, filter_size, stride, input=None):
    if input is None:
      input = self.last_layer

    self.layers[name] = tf.nn.pool(input, [filter_size, filter_size], 'MAX', 'VALID', strides=[stride, stride])
    self.last_layer = self.layers[name]
    return self

  def lrn(self, name, radius, alpha, beta, input=None):
    if input is None:
      input = self.last_layer

    self.layers[name] = tf.nn.lrn(input, depth_radius=radius, alpha=alpha, beta=beta)
    self.last_layer = self.layers[name]
    return self

  def dropout(self, name, input=None):
    if input is None:
      input = self.last_layer

    self.layers[name] = tf.nn.dropout(input, keep_prob=0.5)
    self.last_layer = self.layers[name]
    return self

def main():
  net = MDNet(Config())
  #net.build_trainer(100, 10)
  net.build_generator(10)
  sess = tf.Session()
  net.load('./models/init.npy', sess)

if __name__ == '__main__':
  main()
