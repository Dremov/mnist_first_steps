import tensorflow as tf
import os
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file

tf.reset_default_graph()

saver = tf.placeholder(tf.float32, [128, 784], name='input')

model_path = os.path.join('../save/', 'KerasGlobalAEModel.ckpt')

# Later, launch the model, use the saver to restore variables from disk, and
# do some work with the model.
with tf.Session() as sess:
  # Restore variables from disk.
  print_tensors_in_checkpoint_file(file_name=model_path, tensor_name='', all_tensors=True)

  saver.restore(sess, model_path)
  print("Model restored.")
  # Do some work with the model
  ...