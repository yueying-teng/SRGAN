
import cv2
import os 
import matplotlib.pyplot as plt
import numpy as np
import math
from tensorflow.keras import backend as K
from tensorflow.keras.metrics import MeanAbsoluteError, CategoricalAccuracy
import tensorflow as tf


def get_PSNR(truth, pred):
    """
    PSNR = 20 * log10(MAXp) - 10 * log10(MSE)
    assumes RGB image arrays in range [0, 255]
    """
    truth_data = np.array(truth, dtype='float32')
    pred_data = np.array(pred, dtype='float32')
    diff = pred_data - truth_data
    diff = diff.flatten()
    rmse = math.sqrt(np.mean(diff ** 2.))

    return 20 * math.log10(255.) - 10*math.log10(rmse)


def resolve(model, lr_batch):
    lr_batch = tf.cast(lr_batch, tf.float32)
    sr_batch = model.predict(lr_batch)
    sr_batch = tf.clip_by_value(sr_batch, 0, 255)
    sr_batch = tf.round(sr_batch)
    sr_batch = tf.cast(sr_batch, tf.uint8)
    return sr_batch


def evaluate_psnr(model, dataset):
    psnr_values = []
    for lr, hr in dataset:
        sr = resolve(model, lr)
        for truth, pred in zip(hr, sr):
            psnr_values.append(get_PSNR(truth, pred))
    return tf.reduce_mean(psnr_values)


def evaluate_mae(model, dataset):
    mae_metric = MeanAbsoluteError()
    for lr, hr in dataset:
        sr = resolve(model, lr)
        mae_metric.update_state(hr, sr)
    mae = mae_metric.result()
    mae_metric.reset_states()
    return mae


def evaluate_disc(generator, discriminator, dataset, generated=True):
    disc_acc = CategoricalAccuracy()
    for lr, hr in dataset:
        if generated:
            sr = resolve(generator, lr)
            model_input = tf.cast(sr, tf.float32)
            pred = discriminator.predict(model_input)
            truth = tf.zeros_like(pred)
        else:
            model_input = tf.cast(hr, tf.float32)
            pred = discriminator.predict(model_input)
            truth = tf.ones_like(pred)
        disc_acc.update_state(truth, pred)
    
    acc = disc_acc.result()
    disc_acc.reset_states()
    return acc


def show_batch_imgs(idx, demo_hr_img, demo_lr_img, num_col=2, num_row=1, figsize = (7, 3)):
    plt.figure(figsize=figsize) 
    truth = demo_hr_img[idx]
    lr = demo_lr_img[idx]
    images = [truth, lr]
    
    for i, img in enumerate(images):
        plt.subplot(num_row, num_col, i+1)
        plt.imshow(img)
        plt.xticks([])
        plt.yticks([])
    plt.tight_layout()


def make_len_divisible(img_path):
    hr_img = cv2.imread(img_path)
    h, w, _ = hr_img.shape
    # if any side is not an even number, reduce the hr_img by one row or column
    # so w and h are divisiable in cv2.resize
    if h%2 !=0:
        hr_img = hr_img[:h - 1,:,:]
    if w%2 !=0:
        hr_img = hr_img[:,:w - 1,:]
    
    return hr_img


def process_img(img_path, scale, keep_dim=False):
    hr_img = make_len_divisible(img_path)
    h, w, _ = hr_img.shape

    if not keep_dim:
        lr_img = cv2.resize(hr_img, (w// scale, h// scale))

    return hr_img, lr_img


def get_patch_dim(lr_img, patch_size=200):
    lr_img_shape = lr_img.shape[:2]

    lr_w = np.random.randint(0, lr_img_shape[1] - patch_size + 1)
    lr_h = np.random.randint(0, lr_img_shape[0] - patch_size + 1)

    return lr_w, lr_h


def get_patch_img(lr_w, lr_h, lr_img, sr_img, scale=4, patch_size=50):
    sr_w, sr_h = lr_w*(scale), lr_h*(scale)
    lr_img = lr_img[lr_h:lr_h + patch_size, lr_w:lr_w + patch_size]
    sr_img = sr_img[sr_h:sr_h + patch_size*(scale), sr_w:sr_w + patch_size*(scale)]

    return lr_img, sr_img


def generate_and_save(model, test_img_path, epoch, scale, patch_size=50,
                     dst_dir='/work/logs/srgan_w_edsr_generated_img'):
    # `training` is set to False so that all layers run in inference mode (batchnorm).
    # use the first four images of the first batch from the validation data generator 
    predictions = []
    for path in test_img_path:
        hr_img, lr_img = process_img(path, scale=scale)
        model_input = np.expand_dims(lr_img, 0)
        sr_img = model(model_input, training=False)[0]
        
        # lr_w, lr_h = get_patch_dim(lr_img, patch_size=patch_size)
        lr_w, lr_h = 8, 14
        lr_img, sr_img = get_patch_img(lr_w, lr_h, lr_img, sr_img, scale=scale, patch_size=patch_size)
        predictions.append(sr_img)

    fig = plt.figure(figsize=(20, 15))
    for i in range(len(predictions)):
        plt.subplot(1, len(predictions), i+1)
        plt.imshow(predictions[i]/255)
        plt.axis('off')
    filename = os.path.join(dst_dir, 'generated_imgs_at_epoch_{:04d}.png'.format(epoch))
    plt.savefig(filename)


def write_eval_metrics(tf_board_writer, eval_metric_values, epoch):
    with tf_board_writer.as_default():
        for name, val in eval_metric_values.items():
            tf.summary.scalar(name, val, step=epoch)

def write_losses(tf_board_writer, losses, step):
    with tf_board_writer.as_default():
        for name, metric_obj in losses.items():
            tf.summary.scalar(name, metric_obj.result(), step=step)
