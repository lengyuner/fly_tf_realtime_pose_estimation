








import os
import importlib
import click
import tensorflow as tf
from tensorflow.python.tools import freeze_graph

from train_singlenet_mobilenetv3 import register_tf_netbuilder_extensions
# from util import probe_model
from util import probe_model_singlenet
from models.openpose_singlenet import create_openpose_singlenet


register_tf_netbuilder_extensions()

# load saved model

module = importlib.import_module('models')
create_model_fn = 'create_openpose_singlenet'
create_model = getattr(module, create_model_fn)

path_weights = "output_singlenet/openpose_singlenet"
model = create_model()
model.load_weights(path_weights)

print(model.summary())


ckpt_path = "./ckpt_model/model-20190403-164504.ckpt-205000"
# ckpt-30.data-00000-of-00002
# ckpt-30.data-00001-of-00002
# ckpt-30.index

def export_to_tflite(model, output_path):

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    #converter.experimental_new_converter = True
    #converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
    tflite_model = converter.convert()
    open(output_path, "wb").write(tflite_model)


## https://stackoverflow.com/questions/56766639/how-to-convert-ckpt-to-pb

import os
import tensorflow as tf

trained_checkpoint_prefix = 'models/model.ckpt-49491'
export_dir = os.path.join('export_dir', '0')

graph = tf.Graph()
with tf.compat.v1.Session(graph=graph) as sess:
    # Restore from checkpoint
    loader = tf.compat.v1.train.import_meta_graph(trained_checkpoint_prefix + '.meta')
    loader.restore(sess, trained_checkpoint_prefix)

    # Export checkpoint to SavedModel
    builder = tf.compat.v1.saved_model.builder.SavedModelBuilder(export_dir)
    builder.add_meta_graph_and_variables(sess,
                                         [tf.saved_model.TRAINING, tf.saved_model.SERVING],
                                         strip_default_attrs=True)
    builder.save()




### zhihu
### https://zhuanlan.zhihu.com/p/102302133

# ckpt_filename='ckpt-30.data-00000-of-00002'
import tensorflow as tf
from tensorflow.python.framework import graph_util

def ckpt2pb():
    with tf.Graph().as_default() as graph_old:
        isess = tf.compat.v1.InteractiveSession()
        # isess = tf.Session()
        # ckpt_filename = './model.ckpt'
        ckpt_filename = './tf_ckpts_singlenet/ckpt-30.data-00000-of-00002'
        isess.run(tf.compat.v1.global_variables_initializer())
        saver = tf.compat.v1.train.import_meta_graph(ckpt_filename, clear_devices=True)
        saver.restore(isess, ckpt_filename)

        constant_graph = graph_util.convert_variables_to_constants(isess, isess.graph_def, ["Cls/fc/biases"])
        constant_graph = graph_util.remove_training_nodes(constant_graph)

        with tf.gfile.GFile('./pb_model/model.pb', mode='wb') as f:
            f.write(constant_graph.SerializeToString())





