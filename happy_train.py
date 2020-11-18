










import datetime
from datetime import timedelta
from timeit import default_timer as timer

import tensorflow as tf
from tensorflow.keras.optimizers import Adam

from tf_netbuilder.files import download_checkpoint
from tf_netbuilder_ext.extensions import register_tf_netbuilder_extensions
from models import create_openpose_singlenet
from dataset.generators import get_dataset
from util import plot_to_image, probe_model_singlenet


gpu = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpu[0], True)
gpu = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpu[1], True)

pretrained_mobilenet_v3_url = "https://github.com/michalfaber/tf_netbuilder/releases/download/v1.0/mobilenet_v3_224_1_0.zip"
annot_path_train = '../datasets/fly_2017_dataset/annotations/person_keypoints_train2017.json'
img_dir_train = '../datasets/fly_2017_dataset/train2017/'
annot_path_val = '../datasets/fly_2017_dataset/annotations/person_keypoints_val2017.json'
img_dir_val = '../datasets/fly_2017_dataset/val2017/'
checkpoints_folder = './tf_ckpts_singlenet'
output_weights = 'output_singlenet/openpose_singlenet'
batch_size = 10
lr = 2.5e-5
max_epochs = 30 #300 #TODO(JZ)EPOCH

def eucl_loss(y_true, y_pred):
    return tf.reduce_sum(tf.math.squared_difference(y_pred, y_true))

@tf.function
def train_one_step(model, optimizer, x, y_true):
    with tf.GradientTape() as tape:
        y_pred = model(x)

        losses = [eucl_loss(y_true[0], y_pred[0]),
                  eucl_loss(y_true[0], y_pred[1]),
                  eucl_loss(y_true[0], y_pred[2]),
                  eucl_loss(y_true[1], y_pred[3])
                  ]

        total_loss = tf.reduce_sum(losses)

    grads = tape.gradient(total_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    return losses, total_loss

def train(ds_train, ds_val, model, optimizer, ckpt, last_epoch, last_step, max_epochs, steps_per_epoch):
    train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
    train_loss_heatmap = tf.keras.metrics.Mean('train_loss_heatmap', dtype=tf.float32)
    train_loss_paf = tf.keras.metrics.Mean('train_loss_paf', dtype=tf.float32)

    val_loss = tf.keras.metrics.Mean('val_loss', dtype=tf.float32)
    val_loss_heatmap = tf.keras.metrics.Mean('val_loss_heatmap', dtype=tf.float32)
    val_loss_paf = tf.keras.metrics.Mean('val_loss_paf', dtype=tf.float32)

    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = 'logs_singlenet/gradient_tape/' + current_time + '/train'
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    val_log_dir = 'logs_singlenet/gradient_tape/' + current_time + '/val'
    val_summary_writer = tf.summary.create_file_writer(val_log_dir)

    output_paf_idx = 2
    output_heatmap_idx = 3

    # determine start epoch in case the training has been stopped manually and resumed

    resume = last_step != 0 and (steps_per_epoch - last_step) != 0
    if resume:
        start_epoch = last_epoch
    else:
        start_epoch = last_epoch + 1

    # start processing

    for epoch in range(start_epoch, max_epochs + 1, 1):

        start = timer()

        print("Start processing epoch {}".format(epoch))

        # set the initial step index depending on if you resumed the processing

        if resume:
            step = last_step + 1
            data_iter = ds_train.skip(last_step)
            print(f"Skipping {last_step} steps (May take a few minutes)...")
            resume = False
        else:
            step = 0
            data_iter = ds_train

        # process steps

        for x, y in data_iter:

            step += 1

            losses, total_loss = train_one_step(model, optimizer, x, y)

            train_loss(total_loss)
            train_loss_heatmap(losses[output_heatmap_idx])
            train_loss_paf(losses[output_paf_idx])

            print('step=', step)
            if step % 10 == 0:

                tf.print('Epoch', epoch, f'Step {step}/{steps_per_epoch}', 'Paf1', losses[0], 'Paf2', losses[1], 'Paf3', losses[2],
                         'Heatmap', losses[3], 'Total loss', total_loss)

                with train_summary_writer.as_default():
                    summary_step = (epoch - 1) * steps_per_epoch + step - 1
                    tf.summary.scalar('loss', train_loss.result(), step=summary_step)
                    tf.summary.scalar('loss_heatmap', train_loss_heatmap.result(), step=summary_step)
                    tf.summary.scalar('loss_paf', train_loss_paf.result(), step=summary_step)

            if step % 100 == 0:
                figure = probe_model_singlenet(model, test_img_path="resources/ski_224.jpg")
                with train_summary_writer.as_default():
                    tf.summary.image("Test prediction", plot_to_image(figure), step=step)

            if step % 1000 == 0:
                ckpt.step.assign(step)
                ckpt.epoch.assign(epoch)
                save_path = manager.save()
                print("Saved checkpoint for step {}: {}".format(step, save_path))

            if step >= steps_per_epoch:
                break

        print("Completed epoch {}. Saving weights...".format(epoch))
        model.save_weights(output_weights, overwrite=True)

        # save checkpoint at the end of an epoch

        ckpt.step.assign(step)
        ckpt.epoch.assign(epoch)
        manager.save()

        # reset metrics every epoch

        train_loss.reset_states()
        train_loss_heatmap.reset_states()
        train_loss_paf.reset_states()

        end = timer()

        print("Epoch training time: " + str(timedelta(seconds=end - start)))

        # calculate validation loss

        print("Calculating validation losses...")
        for val_step, (x_val, y_val_true) in enumerate(ds_val):

            if val_step % 1000 == 0:
                print(f"Validation step {val_step} ...")

            y_val_pred = model(x_val)
            losses = [eucl_loss(y_val_true[0], y_val_pred[0]),
                      eucl_loss(y_val_true[0], y_val_pred[1]),
                      eucl_loss(y_val_true[0], y_val_pred[2]),
                      eucl_loss(y_val_true[1], y_val_pred[3])]
            total_loss = tf.reduce_sum(losses)
            val_loss(total_loss)
            val_loss_heatmap(losses[output_heatmap_idx])
            val_loss_paf(losses[output_paf_idx])

        val_loss_res = val_loss.result()
        val_loss_heatmap_res = val_loss_heatmap.result()
        val_loss_paf_res = val_loss_paf.result()

        print(f'Validation losses for epoch: {epoch} : Loss paf {val_loss_paf_res}, Loss heatmap '
              f'{val_loss_heatmap_res}, Total loss {val_loss_res}')

        with val_summary_writer.as_default():
            tf.summary.scalar('val_loss', val_loss_res, step=epoch)
            tf.summary.scalar('val_loss_heatmap', val_loss_heatmap_res, step=epoch)
            tf.summary.scalar('val_loss_paf', val_loss_paf_res, step=epoch)
        val_loss.reset_states()
        val_loss_heatmap.reset_states()
        val_loss_paf.reset_states()





# registering custom blocks types
print('############## begin ###############\n'*10)
register_tf_netbuilder_extensions()

# loading datasets
print('############## begin loading datasets ###############\n' * 10)
ds_train, ds_train_size = get_dataset(annot_path_train, img_dir_train, batch_size)
ds_val, ds_val_size = get_dataset(annot_path_val, img_dir_val, batch_size, strict=True)
print(f"Training samples: {ds_train_size} , Validation samples: {ds_val_size}")

steps_per_epoch = ds_train_size // batch_size
steps_per_epoch_val = ds_val_size // batch_size

print('############## begin creating model ###############\n' * 10)
# creating model, optimizers etc
model = create_openpose_singlenet(pretrained=False)
optimizer = Adam(lr)
# loading previous state if required
ckpt = tf.train.Checkpoint(step=tf.Variable(0), epoch=tf.Variable(0), optimizer=optimizer, net=model)
manager = tf.train.CheckpointManager(ckpt, checkpoints_folder, max_to_keep=3)
ckpt.restore(manager.latest_checkpoint)
last_step = int(ckpt.step)
last_epoch = int(ckpt.epoch)


# path_download = download_checkpoint(pretrained_mobilenet_v3_url)
path_download = 'C:\\Users\\ps\\.cache\\tf_netbuilder\\' \
                'checkpoints\\mobilenet_v3_224_1_0\\mobilenet_v3_224_1_0'
model.load_weights(path_download).expect_partial()
print(model.summary())



train(ds_train, ds_val, model, optimizer, ckpt, last_epoch, last_step,
      max_epochs, steps_per_epoch)


#
# '''
#
# val_get_dataset=0
# if val_get_dataset==1:
#     ds_train, ds_train_size = get_dataset(annot_path_train, img_dir_train, batch_size)
#     # 这里有问题，等下来改
# else:
#     import tensorflow as tf
#     from dataset.dataflows import get_dataflow, get_dataflow_vgg
#
#     annot_path = annot_path_train
#     img_dir = img_dir_train
#     x_size = 224
#     y_size = 28
#
#
#     def gen(df):
#         def f():
#             for i in df:
#                 yield tuple(i)
#
#         return f
#
#
#     # 这里有问题，等下来改
# strict = False
# val_get_dataflow = 0
# if val_get_dataflow == 1:
#         df, size = get_dataflow(
#             annot_path=annot_path,
#             img_dir=img_dir,
#             strict=strict,
#             x_size=x_size,
#             y_size=y_size
#         )
# else:
#
#     import os
#     import cv2
#     import numpy as np
#     import functools
#
#     from tensorpack.dataflow import MultiProcessMapDataZMQ, TestDataSpeed
#     from tensorpack.dataflow.common import MapData
#
#     from dataset.augmentors import CropAug, FlipAug, ScaleAug, RotateAug, ResizeAug
#     from dataset.base_dataflow import CocoDataFlow, JointsLoader
#     from dataset.dataflow_steps import create_all_mask, augment, read_img, apply_mask, gen_mask
#     from dataset.label_maps import create_heatmap, create_paf
#
#     # coco_crop_size = 368
#     coco_crop_size = 224
#     # TODO(JZ)crop_size
#
#     augmentors = [
#         ScaleAug(scale_min=0.5,
#                  scale_max=1.1,
#                  target_dist=0.6,
#                  interp=cv2.INTER_CUBIC),
#
#         RotateAug(rotate_max_deg=40,
#                   interp=cv2.INTER_CUBIC,
#                   border=cv2.BORDER_CONSTANT,
#                   border_value=(128, 128, 128), mask_border_val=1),
#
#         CropAug(coco_crop_size, coco_crop_size, center_perterb_max=40, border_value=128,
#                 mask_border_val=1),
#
#         FlipAug(num_parts=5, prob=0.5),
#         # TODO(JZ)FlipAug
#         ResizeAug(x_size, x_size)
#
#     ]
#
#     # prepare augment function
#
#     augment_func = functools.partial(augment,
#                                      augmentors=augmentors)
#
#
#     # prepare building sample function
#
#     def build_sample(components, y_size):
#         """
#         Builds a sample for a model.
#
#         :param components: components
#         :return: list of final components of a sample.
#         """
#         img = components[10]
#         aug_joints = components[13]
#
#         heatmap = create_heatmap(JointsLoader.num_joints_and_bkg, y_size, y_size,
#                                  aug_joints, 5.0, stride=8)
#
#         pafmap = create_paf(JointsLoader.num_connections, y_size, y_size,
#                             aug_joints, 0.8, stride=8)
#
#         return [img,
#                 pafmap,
#                 heatmap]
#
#
#     build_sample_func = functools.partial(build_sample,
#                                           y_size=y_size)
#
#     # build the dataflow
#
#     df = CocoDataFlow((coco_crop_size, coco_crop_size), annot_path, img_dir, None)
#
# val_df_prepare = 1
# if val_df_prepare == 1:
#     #
#     df.prepare()
#     # 这里有问题，等下来改
# else:
#     import os
#     import numpy as np
#
#     from scipy.spatial.distance import cdist
#     from pycocotools.coco import COCO
#     from tensorpack.dataflow.base import RNGDataFlow
#
#
#     class Meta(object):
#         """
#         Metadata representing a single data point for training.
#         """
#         __slots__ = (
#             'img_path',
#             'height',
#             'width',
#             'center',
#             'bbox',
#             'area',
#             'num_keypoints',
#             'masks_segments',
#             'scale',
#             'all_joints',
#             'img',
#             'mask',
#             'aug_center',
#             'aug_joints')
#
#         def __init__(self, img_path, height, width, center, bbox,
#                      area, scale, num_keypoints):
#             self.img_path = img_path
#             self.height = height
#             self.width = width
#             self.center = center
#             self.bbox = bbox
#             self.area = area
#             self.scale = scale
#             self.num_keypoints = num_keypoints
#
#             # updated after iterating over all persons
#             self.masks_segments = None
#             self.all_joints = None
#
#             # updated during augmentation
#             self.img = None
#             self.mask = None
#             self.aug_center = None
#             self.aug_joints = None
#
#
#     self1 = df
#
#     if self1.select_ids:
#         ids = self1.select_ids
#     else:
#         ids = list(self1.coco.imgs.keys())
#
#     for i, img_id in enumerate(ids):
#         # for i, img_id in enumerate(ids[0:32]):
#         # for i, img_id in enumerate(ids[32:33]):
#         img_meta = self1.coco.imgs[img_id]
#
#         # load annotations
#
#         img_id = img_meta['id']
#         img_file = img_meta['file_name']
#         h, w = img_meta['height'], img_meta['width']
#         img_path = os.path.join(self1.img_dir, img_file)
#         ann_ids = self1.coco.getAnnIds(imgIds=img_id)
#         # print(ann_ids)
#         anns = self1.coco.loadAnns(ann_ids)
#         # print(anns)
#
#         total_keypoints = sum([ann.get('num_keypoints', 0) for ann in anns])
#         if total_keypoints == 0:
#             continue
#
#         persons = []
#         prev_center = []
#         masks = []
#         keypoints = []
#
#         # sort from the biggest person to the smallest one
#
#         persons_ids = np.argsort([-a['area'] for a in anns], kind='mergesort')
#
#         for id in list(persons_ids):
#             print(id)
#             id = 0
#             person_meta = anns[id]
#
#             if person_meta["iscrowd"]:
#                 masks.append(self1.coco.annToRLE(person_meta))
#                 continue
#
#             # skip this person if parts number is too low or if
#             # segmentation area is too small
#
#             if person_meta["num_keypoints"] < 5 or person_meta["area"] < 32 * 32:
#                 # TODO(JZ)
#                 masks.append(self1.coco.annToRLE(person_meta))
#                 continue
#
#             person_center = [person_meta["bbox"][0] + person_meta["bbox"][2] / 2,
#                              person_meta["bbox"][1] + person_meta["bbox"][3] / 2]
#
#             # skip this person if the distance to existing person is too small
#
#             too_close = False
#             # prev_center=【】
#             for pc in prev_center:
#                 a = np.expand_dims(pc[:2], axis=0)
#                 b = np.expand_dims(person_center, axis=0)
#                 dist = cdist(a, b)[0]
#                 if dist < pc[2] * 0.3:
#                     too_close = True
#                     break
#
#             if too_close:
#                 # add mask of this person. we don't want to show the network
#                 # unlabeled people
#                 masks.append(self1.coco.annToRLE(person_meta))
#                 continue
#
#             pers = Meta(
#                 img_path=img_path,
#                 height=h,
#                 width=w,
#                 center=np.expand_dims(person_center, axis=0),
#                 bbox=person_meta["bbox"],
#                 area=person_meta["area"],
#                 scale=person_meta["bbox"][3] / self1.target_size[0],
#                 num_keypoints=person_meta["num_keypoints"])
#
#             keypoints.append(person_meta["keypoints"])
#             persons.append(pers)
#             prev_center.append(np.append(person_center, max(person_meta["bbox"][2],
#                                                             person_meta["bbox"][3])))
#
#         # use only main persons and ignore the smaller ones
#         if len(persons) > 0:
#             main_person = persons[0]
#             main_person.masks_segments = masks
#             main_person.all_joints = JointsLoader.from_coco_keypoints(keypoints, w, h)
#             self1.all_meta.append(main_person)
#         print("Loading image annot {}/{}".format(i, len(ids)))
#         # if i % 1000 == 0:
#         #     print("Loading image annot {}/{}".format(i, len(ids)))
#
#     df = self1
# if 1<0:
#     size = df.size()
#
#     df = MapData(df, read_img)
#
#     df = MapData(df, augment_func)
#
#     from tensorpack.dataflow import MultiThreadMapData
#
#     # TODO(ZMQ
#     df = MultiThreadMapData(df, 4, build_sample_func,
#                             buffer_size=200, strict=strict)
#
#     df.reset_state()
#     #  reset_state
#     ds = tf.data.Dataset.from_generator(
#         gen(df), (tf.float32, tf.float32, tf.float32),
#         output_shapes=(
#             tf.TensorShape([x_size, x_size, 3]),
#             tf.TensorShape([y_size, y_size, 12]),
#             tf.TensorShape([y_size, y_size, 6])  # TODO(jz)
#
#         )
#     )
#
#     ds = ds.map(lambda x0, x1, x2: (x0, (x1, x2)))
#     ds = ds.batch(batch_size)
#
#     ds_train, ds_train_size = ds, size
#
#     # ds_train_size=123.134123428900888
#     print(f"Training samples: {ds_train_size} ")
# '''






# train(ds_train, ds_val, model, optimizer, ckpt, last_epoch, last_step,
#       max_epochs, steps_per_epoch)

# basepath = 'C:\\Users\\ps\\.cache\\tf_netbuilder\\checkpoints\\'
# path = basepath + 'mobilenet_v3_224_1_0\\'+ 'mobilenet_v3_224_1_0\\'
# model.load_weights(path).expect_partial()
# training loop
# basepath = 'C:/Users/dongj/Desktop/courses/AI/Experiment/xihu/keras_Realtime_Multi-Person_Pose_Estimation/'
# weights_path = basepath + 'model/keras/model.h5'




# path = r'C:\Users\ps\.cache\tf_netbuilder\checkpoints\mobilenet_v3_224_1_0\mobilenet_v3_224_1_0'
# # model.load_weights(path).expect_partial()
#
#
# '''
#
#
#
#
#
#
#
#
#
#
# import datetime
# from datetime import timedelta
# from timeit import default_timer as timer
#
# import tensorflow as tf
# from tensorflow.keras.optimizers import Adam
#
# from tf_netbuilder.files import download_checkpoint
# from tf_netbuilder_ext.extensions import register_tf_netbuilder_extensions
# from models import create_openpose_singlenet
# from dataset.generators import get_dataset
# from util import plot_to_image, probe_model_singlenet
#
#
# pretrained_mobilenet_v3_url = "https://github.com/michalfaber/tf_netbuilder/releases/download/v1.0/mobilenet_v3_224_1_0.zip"
# annot_path_train = '../datasets/fly_2017_dataset/annotations/person_keypoints_train2017.json'
# img_dir_train = '../datasets/fly_2017_dataset/train2017/'
# annot_path_val = '../datasets/fly_2017_dataset/annotations/person_keypoints_val2017.json'
# img_dir_val = '../datasets/fly_2017_dataset/val2017/'
# checkpoints_folder = './tf_ckpts_singlenet'
# output_weights = 'output_singlenet/openpose_singlenet'
# batch_size = 10
# lr = 2.5e-5
# max_epochs = 30 #300 #TODO(JZ)EPOCH
#
# def eucl_loss(y_true, y_pred):
#     return tf.reduce_sum(tf.math.squared_difference(y_pred, y_true))
#
# @tf.function
# def train_one_step(model, optimizer, x, y_true):
#     with tf.GradientTape() as tape:
#         y_pred = model(x)
#
#         losses = [eucl_loss(y_true[0], y_pred[0]),
#                   eucl_loss(y_true[0], y_pred[1]),
#                   eucl_loss(y_true[0], y_pred[2]),
#                   eucl_loss(y_true[1], y_pred[3])
#                   ]
#
#         total_loss = tf.reduce_sum(losses)
#
#     grads = tape.gradient(total_loss, model.trainable_variables)
#     optimizer.apply_gradients(zip(grads, model.trainable_variables))
#
#     return losses, total_loss
#
# def train(ds_train, ds_val, model, optimizer, ckpt, last_epoch, last_step, max_epochs, steps_per_epoch):
#     train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
#     train_loss_heatmap = tf.keras.metrics.Mean('train_loss_heatmap', dtype=tf.float32)
#     train_loss_paf = tf.keras.metrics.Mean('train_loss_paf', dtype=tf.float32)
#
#     val_loss = tf.keras.metrics.Mean('val_loss', dtype=tf.float32)
#     val_loss_heatmap = tf.keras.metrics.Mean('val_loss_heatmap', dtype=tf.float32)
#     val_loss_paf = tf.keras.metrics.Mean('val_loss_paf', dtype=tf.float32)
#
#     current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
#     train_log_dir = 'logs_singlenet/gradient_tape/' + current_time + '/train'
#     train_summary_writer = tf.summary.create_file_writer(train_log_dir)
#     val_log_dir = 'logs_singlenet/gradient_tape/' + current_time + '/val'
#     val_summary_writer = tf.summary.create_file_writer(val_log_dir)
#
#     output_paf_idx = 2
#     output_heatmap_idx = 3
#
#     # determine start epoch in case the training has been stopped manually and resumed
#
#     resume = last_step != 0 and (steps_per_epoch - last_step) != 0
#     if resume:
#         start_epoch = last_epoch
#     else:
#         start_epoch = last_epoch + 1
#
#     # start processing
#
#     for epoch in range(start_epoch, max_epochs + 1, 1):
#
#         start = timer()
#
#         print("Start processing epoch {}".format(epoch))
#
#         # set the initial step index depending on if you resumed the processing
#
#         if resume:
#             step = last_step + 1
#             data_iter = ds_train.skip(last_step)
#             print(f"Skipping {last_step} steps (May take a few minutes)...")
#             resume = False
#         else:
#             step = 0
#             data_iter = ds_train
#
#         # process steps
#
#         for x, y in data_iter:
#
#             step += 1
#
#             losses, total_loss = train_one_step(model, optimizer, x, y)
#
#             train_loss(total_loss)
#             train_loss_heatmap(losses[output_heatmap_idx])
#             train_loss_paf(losses[output_paf_idx])
#
#             print('step=', step)
#             if step % 10 == 0:
#
#                 tf.print('Epoch', epoch, f'Step {step}/{steps_per_epoch}', 'Paf1', losses[0], 'Paf2', losses[1], 'Paf3', losses[2],
#                          'Heatmap', losses[3], 'Total loss', total_loss)
#
#                 with train_summary_writer.as_default():
#                     summary_step = (epoch - 1) * steps_per_epoch + step - 1
#                     tf.summary.scalar('loss', train_loss.result(), step=summary_step)
#                     tf.summary.scalar('loss_heatmap', train_loss_heatmap.result(), step=summary_step)
#                     tf.summary.scalar('loss_paf', train_loss_paf.result(), step=summary_step)
#
#             if step % 100 == 0:
#                 figure = probe_model_singlenet(model, test_img_path="resources/ski_224.jpg")
#                 with train_summary_writer.as_default():
#                     tf.summary.image("Test prediction", plot_to_image(figure), step=step)
#
#             if step % 1000 == 0:
#                 ckpt.step.assign(step)
#                 ckpt.epoch.assign(epoch)
#                 save_path = manager.save()
#                 print("Saved checkpoint for step {}: {}".format(step, save_path))
#
#             if step >= steps_per_epoch:
#                 break
#
#         print("Completed epoch {}. Saving weights...".format(epoch))
#         model.save_weights(output_weights, overwrite=True)
#
#         # save checkpoint at the end of an epoch
#
#         ckpt.step.assign(step)
#         ckpt.epoch.assign(epoch)
#         manager.save()
#
#         # reset metrics every epoch
#
#         train_loss.reset_states()
#         train_loss_heatmap.reset_states()
#         train_loss_paf.reset_states()
#
#         end = timer()
#
#         print("Epoch training time: " + str(timedelta(seconds=end - start)))
#
#         # calculate validation loss
#
#         print("Calculating validation losses...")
#         for val_step, (x_val, y_val_true) in enumerate(ds_val):
#
#             if val_step % 1000 == 0:
#                 print(f"Validation step {val_step} ...")
#
#             y_val_pred = model(x_val)
#             losses = [eucl_loss(y_val_true[0], y_val_pred[0]),
#                       eucl_loss(y_val_true[0], y_val_pred[1]),
#                       eucl_loss(y_val_true[0], y_val_pred[2]),
#                       eucl_loss(y_val_true[1], y_val_pred[3])]
#             total_loss = tf.reduce_sum(losses)
#             val_loss(total_loss)
#             val_loss_heatmap(losses[output_heatmap_idx])
#             val_loss_paf(losses[output_paf_idx])
#
#         val_loss_res = val_loss.result()
#         val_loss_heatmap_res = val_loss_heatmap.result()
#         val_loss_paf_res = val_loss_paf.result()
#
#         print(f'Validation losses for epoch: {epoch} : Loss paf {val_loss_paf_res}, Loss heatmap '
#               f'{val_loss_heatmap_res}, Total loss {val_loss_res}')
#
#         with val_summary_writer.as_default():
#             tf.summary.scalar('val_loss', val_loss_res, step=epoch)
#             tf.summary.scalar('val_loss_heatmap', val_loss_heatmap_res, step=epoch)
#             tf.summary.scalar('val_loss_paf', val_loss_paf_res, step=epoch)
#         val_loss.reset_states()
#         val_loss_heatmap.reset_states()
#         val_loss_paf.reset_states()
#
#
#
# # registering custom blocks types
# print('############## begin ###############\n'*10)
# register_tf_netbuilder_extensions()
#
#
#
# val_get_dataset=0
# if val_get_dataset==1:
#     ds_train, ds_train_size = get_dataset(annot_path_train, img_dir_train, batch_size)
#     # 这里有问题，等下来改
# else:
#     import tensorflow as tf
#     from dataset.dataflows import get_dataflow, get_dataflow_vgg
#
#     annot_path = annot_path_train
#     img_dir = img_dir_train
#     x_size = 224
#     y_size = 28
#
#
#     def gen(df):
#         def f():
#             for i in df:
#                 yield tuple(i)
#
#         return f
#
#
#     # 这里有问题，等下来改
#     strict = False
#     val_get_dataflow = 0
#     if val_get_dataflow == 1:
#         df, size = get_dataflow(
#             annot_path=annot_path,
#             img_dir=img_dir,
#             strict=strict,
#             x_size=x_size,
#             y_size=y_size
#         )
#     else:
#         import os
#         import cv2
#         import numpy as np
#         import functools
#
#         from tensorpack.dataflow import MultiProcessMapDataZMQ, TestDataSpeed
#         from tensorpack.dataflow.common import MapData
#
#         from dataset.augmentors import CropAug, FlipAug, ScaleAug, RotateAug, ResizeAug
#         from dataset.base_dataflow import CocoDataFlow, JointsLoader
#         from dataset.dataflow_steps import create_all_mask, augment, read_img, apply_mask, gen_mask
#         from dataset.label_maps import create_heatmap, create_paf
#
#         coco_crop_size = 368
#
#         augmentors = [
#             ScaleAug(scale_min=0.5,
#                      scale_max=1.1,
#                      target_dist=0.6,
#                      interp=cv2.INTER_CUBIC),
#
#             RotateAug(rotate_max_deg=40,
#                       interp=cv2.INTER_CUBIC,
#                       border=cv2.BORDER_CONSTANT,
#                       border_value=(128, 128, 128), mask_border_val=1),
#
#             CropAug(coco_crop_size, coco_crop_size, center_perterb_max=40, border_value=128,
#                     mask_border_val=1),
#
#             FlipAug(num_parts=18, prob=0.5),
#
#             ResizeAug(x_size, x_size)
#
#         ]
#
#         # prepare augment function
#
#         augment_func = functools.partial(augment,
#                                          augmentors=augmentors)
#
#
#         # prepare building sample function
#
#         def build_sample(components, y_size):
#             """
#             Builds a sample for a model.
#
#             :param components: components
#             :return: list of final components of a sample.
#             """
#             img = components[10]
#             aug_joints = components[13]
#
#             heatmap = create_heatmap(JointsLoader.num_joints_and_bkg, y_size, y_size,
#                                      aug_joints, 5.0, stride=8)
#
#             pafmap = create_paf(JointsLoader.num_connections, y_size, y_size,
#                                 aug_joints, 0.8, stride=8)
#
#             return [img,
#                     pafmap,
#                     heatmap]
#
#
#         build_sample_func = functools.partial(build_sample,
#                                               y_size=y_size)
#
#         # build the dataflow
#
#         df = CocoDataFlow((coco_crop_size, coco_crop_size), annot_path, img_dir, None)
#
#         val_df_prepare = 1
#         if val_df_prepare == 1:
#             #
#             df.prepare()
#             # 这里有问题，等下来改
#         else:
#
#             import os
#             import numpy as np
#
#             from scipy.spatial.distance import cdist
#             from pycocotools.coco import COCO
#             from tensorpack.dataflow.base import RNGDataFlow
#
#
#             class Meta(object):
#                 """
#                 Metadata representing a single data point for training.
#                 """
#                 __slots__ = (
#                     'img_path',
#                     'height',
#                     'width',
#                     'center',
#                     'bbox',
#                     'area',
#                     'num_keypoints',
#                     'masks_segments',
#                     'scale',
#                     'all_joints',
#                     'img',
#                     'mask',
#                     'aug_center',
#                     'aug_joints')
#
#                 def __init__(self, img_path, height, width, center, bbox,
#                              area, scale, num_keypoints):
#                     self.img_path = img_path
#                     self.height = height
#                     self.width = width
#                     self.center = center
#                     self.bbox = bbox
#                     self.area = area
#                     self.scale = scale
#                     self.num_keypoints = num_keypoints
#
#                     # updated after iterating over all persons
#                     self.masks_segments = None
#                     self.all_joints = None
#
#                     # updated during augmentation
#                     self.img = None
#                     self.mask = None
#                     self.aug_center = None
#                     self.aug_joints = None
#
#
#             self1 = df
#
#             if self1.select_ids:
#                 ids = self1.select_ids
#             else:
#                 ids = list(self1.coco.imgs.keys())
#
#             for i, img_id in enumerate(ids):
#                 # for i, img_id in enumerate(ids[0:32]):
#                 # for i, img_id in enumerate(ids[32:33]):
#                 img_meta = self1.coco.imgs[img_id]
#
#                 # load annotations
#
#
#                 img_id = img_meta['id']
#                 img_file = img_meta['file_name']
#                 h, w = img_meta['height'], img_meta['width']
#                 img_path = os.path.join(self1.img_dir, img_file)
#                 ann_ids = self1.coco.getAnnIds(imgIds=img_id)
#                 # print(ann_ids)
#                 anns = self1.coco.loadAnns(ann_ids)
#                 # print(anns)
#
#                 total_keypoints = sum([ann.get('num_keypoints', 0) for ann in anns])
#                 if total_keypoints == 0:
#                     continue
#
#                 persons = []
#                 prev_center = []
#                 masks = []
#                 keypoints = []
#
#                 # sort from the biggest person to the smallest one
#
#                 persons_ids = np.argsort([-a['area'] for a in anns], kind='mergesort')
#
#                 for id in list(persons_ids):
#                     print(id)
#                     id = 0
#                     person_meta = anns[id]
#
#                     if person_meta["iscrowd"]:
#                         masks.append(self1.coco.annToRLE(person_meta))
#                         continue
#
#                     # skip this person if parts number is too low or if
#                     # segmentation area is too small
#
#                     if person_meta["num_keypoints"] < 5 or person_meta["area"] < 32 * 32:
#                         # TODO(JZ)
#                         masks.append(self1.coco.annToRLE(person_meta))
#                         continue
#
#                     person_center = [person_meta["bbox"][0] + person_meta["bbox"][2] / 2,
#                                      person_meta["bbox"][1] + person_meta["bbox"][3] / 2]
#
#                     # skip this person if the distance to existing person is too small
#
#                     too_close = False
#                     # prev_center=【】
#                     for pc in prev_center:
#                         a = np.expand_dims(pc[:2], axis=0)
#                         b = np.expand_dims(person_center, axis=0)
#                         dist = cdist(a, b)[0]
#                         if dist < pc[2] * 0.3:
#                             too_close = True
#                             break
#
#                     if too_close:
#                         # add mask of this person. we don't want to show the network
#                         # unlabeled people
#                         masks.append(self1.coco.annToRLE(person_meta))
#                         continue
#
#                     pers = Meta(
#                         img_path=img_path,
#                         height=h,
#                         width=w,
#                         center=np.expand_dims(person_center, axis=0),
#                         bbox=person_meta["bbox"],
#                         area=person_meta["area"],
#                         scale=person_meta["bbox"][3] / self1.target_size[0],
#                         num_keypoints=person_meta["num_keypoints"])
#
#                     keypoints.append(person_meta["keypoints"])
#                     persons.append(pers)
#                     prev_center.append(np.append(person_center, max(person_meta["bbox"][2],
#                                                                     person_meta["bbox"][3])))
#
#                 # use only main persons and ignore the smaller ones
#                 if len(persons) > 0:
#                     main_person = persons[0]
#                     main_person.masks_segments = masks
#                     main_person.all_joints = JointsLoader.from_coco_keypoints(keypoints, w, h)
#                     self1.all_meta.append(main_person)
#                 print("Loading image annot {}/{}".format(i, len(ids)))
#                 # if i % 1000 == 0:
#                 #     print("Loading image annot {}/{}".format(i, len(ids)))
#
#             df = self1
#
#         size = df.size()
#
#         df = MapData(df, read_img)
#
#         df = MapData(df, augment_func)
#
#         from tensorpack.dataflow import MultiThreadMapData
#
#         #TODO(ZMQ
#         df = MultiThreadMapData(df, 4, build_sample_func,
#                                 buffer_size=200, strict=strict)
#
#     df.reset_state()
#     #  reset_state
#     ds = tf.data.Dataset.from_generator(
#         gen(df), (tf.float32, tf.float32, tf.float32),
#         output_shapes=(
#             tf.TensorShape([x_size, x_size, 3]),
#             tf.TensorShape([y_size, y_size, 12]),
#             tf.TensorShape([y_size, y_size, 6])  # TODO(jz)
#
#         )
#     )
#
#     ds = ds.map(lambda x0, x1, x2: (x0, (x1, x2)))
#     ds = ds.batch(batch_size)
#
#     ds_train, ds_train_size = ds, size
#
# # ds_train_size=123.134123428900888
# print(f"Training samples: {ds_train_size} ")
#
#
# ds_train, ds_train_size=df, size
# # print(ds_train, ds_train_size)
# # print(ds_train, ds_train_size)
#
#
#
# # loading datasets
# print('############## begin loading datasets ###############\n' * 10)
# # ds_train, ds_train_size = get_dataset(annot_path_train, img_dir_train, batch_size)
# ds_val, ds_val_size = get_dataset(annot_path_val, img_dir_val, batch_size, strict=True)
# print(f"Training samples: {ds_train_size} , Validation samples: {ds_val_size}")
#
# steps_per_epoch = ds_train_size // batch_size
# steps_per_epoch_val = ds_val_size // batch_size
#
# print('############## begin creating model ###############\n' * 10)
#
# # creating model, optimizers etc
#
# model = create_openpose_singlenet(pretrained=False)
# optimizer = Adam(lr)
#
# # loading previous state if required
#
# ckpt = tf.train.Checkpoint(step=tf.Variable(0), epoch=tf.Variable(0), optimizer=optimizer, net=model)
# manager = tf.train.CheckpointManager(ckpt, checkpoints_folder, max_to_keep=3)
# ckpt.restore(manager.latest_checkpoint)
# last_step = int(ckpt.step)
# last_epoch = int(ckpt.epoch)
#
# # if manager.latest_checkpoint:
# #     print(f"Restored from {manager.latest_checkpoint}")
# #     print(f"Resumed from epoch {last_epoch}, step {last_step}")
# # else:
# #     print("Initializing from scratch.")
# path_download = download_checkpoint(pretrained_mobilenet_v3_url)
#     # path = r'C:\Users\ps\.cache\tf_netbuilder\checkpoints\mobilenet_v3_224_1_0.zip'
# model.load_weights(path_download).expect_partial()
# print(model.summary())
#
# train(ds_train, ds_train, model, optimizer, ckpt, last_epoch, last_step,
#       max_epochs, steps_per_epoch)
#
#
# # train(ds_train, ds_val, model, optimizer, ckpt, last_epoch, last_step,
# #       max_epochs, steps_per_epoch)
#
# # basepath = 'C:\\Users\\ps\\.cache\\tf_netbuilder\\checkpoints\\'
# # path = basepath + 'mobilenet_v3_224_1_0\\'+ 'mobilenet_v3_224_1_0\\'
# # model.load_weights(path).expect_partial()
# # training loop
# # basepath = 'C:/Users/dongj/Desktop/courses/AI/Experiment/xihu/keras_Realtime_Multi-Person_Pose_Estimation/'
# # weights_path = basepath + 'model/keras/model.h5'
#
#
#
#
# # path = r'C:\Users\ps\.cache\tf_netbuilder\checkpoints\mobilenet_v3_224_1_0\mobilenet_v3_224_1_0'
# # model.load_weights(path).expect_partial()
#
#
# '''

