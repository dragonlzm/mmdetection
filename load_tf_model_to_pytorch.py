import tensorflow.compat.v1 as tf

tf.load_op_library()

with tf.Session() as sess:
  new_saver = tf.train.import_meta_graph('model.ckpt-180000.meta')
  new_saver.restore(sess, tf.train.latest_checkpoint('./'))
  
  
  
tf.all_variables()