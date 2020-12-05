'''Single neuron neural network with TensorBoard annotations'''

from __future__ import print_function
import numpy as np
import tensorflow as tf

# Start with a clean environment each run
log_dir = '/tmp/tensorflow/nn/train' # TensorBoard logs saved here 
if tf.gfile.Exists(log_dir):
  tf.gfile.DeleteRecursively(log_dir)
tf.gfile.MakeDirs(log_dir)
tf.reset_default_graph()

def variable_summaries(name, var):
  """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
  with tf.name_scope('weights'):
    with tf.name_scope('summaries'):
      mean = tf.reduce_mean(var)
      tf.summary.scalar('mean', mean)
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
      tf.summary.scalar('stddev', stddev)
      tf.summary.scalar('max', tf.reduce_max(var))
      tf.summary.scalar('min', tf.reduce_min(var))
      tf.summary.histogram('histogram', var)

# Set up sample points perturbed away from the ideal linear relationship 
# y = 0.5*x + 2.5
num_examples = 60
points = np.array([np.linspace(-1, 5, num_examples),
  np.linspace(2, 5, num_examples)])
points += np.random.randn(2, num_examples)
x, y = points
# Include a 1 to use as the bias input for neurons
x_with_bias = np.array([(1., d) for d in x]).astype(np.float32)

# Training parameters
training_steps = 100
learning_rate = 0.001
losses = []

with tf.Session() as sess:
  # Set up all the tensors, variables, and operations.
  input = tf.constant(x_with_bias)
  target = tf.constant(np.transpose([y]).astype(np.float32))
  # Initialize weights with small random values
  weights = tf.Variable(tf.random_normal([2, 1], 0, 0.1))

  tf.global_variables_initializer().run()

  # Calculate the current prediction error
  y_predicted = tf.matmul(input, weights)
  y_error = tf.subtract(y_predicted, target)

  # Compute the L2 loss function of the error
  loss = tf.nn.l2_loss(y_error)
 
  # Train the network using an optimizer that minimizes the loss function
  update_weights = tf.train.GradientDescentOptimizer(
  learning_rate).minimize(loss)

  # Add summary operations
  variable_summaries('weights', weights)
  tf.summary.histogram('y_error', y_error)
  tf.summary.scalar('loss', loss)
  merged = tf.summary.merge_all()
  summary_writer = tf.summary.FileWriter(log_dir, graph=tf.get_default_graph())
 
  for i in range(training_steps):
    # Repeatedly run the summary and training operations
    summary, updates = sess.run([merged, update_weights])
    summary_writer.add_summary(summary, i)

  summary_writer.close()

print('Complete')