import os
import time
import datetime
import tensorflow as tf
from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.losses import MeanAbsoluteError
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import Mean
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import PiecewiseConstantDecay
from SRGAN import models 
from SRGAN import utils
import numpy as np
## https://github.com/krasserm/super-resolution


class Trainer:
    def __init__(self,
                 model,
                 loss,
                 learning_rate,
                 checkpoint_dir='./ckpt/edsr'):

        self.now = None
        self.loss = loss
        os.makedirs(checkpoint_dir, exist_ok=True)
        self.checkpoint = tf.train.Checkpoint(step=tf.Variable(0),
                                              psnr=tf.Variable(-1.0),
                                              optimizer=Adam(learning_rate),
                                              model=model)
        self.checkpoint_manager = tf.train.CheckpointManager(checkpoint=self.checkpoint,
                                                             directory=checkpoint_dir,
                                                             max_to_keep=3)
        self.log_var = []
        self.restore()

    @property
    def model(self):
        return self.checkpoint.model

    def get_log(self):
        return self.log_var
    
    def train(self, train_dataset, valid_dataset, steps, evaluate_every=1000, save_best_only=False):
        loss_mean = Mean()

        ckpt_mgr = self.checkpoint_manager
        ckpt = self.checkpoint
        self.now = time.perf_counter()

        restored_steps = ckpt.step.numpy()
        restored_epoch = restored_steps // len(train_dataset)
        total_epochs = steps // len(train_dataset)
        new_step_start = restored_steps % len(train_dataset)
        self.log_var.append('step,train_loss,val_loss,val_PSNR,duration')
        
        for epoch in range(restored_epoch, total_epochs):
            for _, data in enumerate(train_dataset):
                step = ckpt.step.numpy()
                if step < new_step_start and restored_steps !=0:
                    continue
                ckpt.step.assign_add(1)
                # print(step)
                lr, hr = data[0], data[1]
                loss = self.train_step(lr, hr)
                loss_mean(loss)

                if step % evaluate_every == 0:
                    loss_value = loss_mean.result()
                    loss_mean.reset_states()
                    # print(step, loss_value)
                    # Compute PSNR on validation dataset
                    psnr_value = self.evaluate_psnr(valid_dataset)
                    val_loss = self.evaluate_mae(valid_dataset)
                    duration = time.perf_counter() - self.now
                    print(f'{step}/{steps}: train_loss = {loss_value.numpy():.3f}, val_loss = {val_loss.numpy():.3f}, val_PSNR = {psnr_value.numpy():3f} ({duration:.2f}s)')
                    to_log = f'{step}/{steps}, {loss_value.numpy():.3f}, {val_loss.numpy():.3f}, {psnr_value.numpy():3f}, {duration:.2f}s'
                    self.log_var.append(to_log)
                     
                    if save_best_only and psnr_value <= ckpt.psnr:
                        self.now = time.perf_counter()
                        # skip saving checkpoint, no PSNR improvement
                        continue

                    ckpt.psnr = psnr_value
                    ckpt_mgr.save()
                    self.now = time.perf_counter()

    @tf.function
    def train_step(self, lr, hr):
        with tf.GradientTape() as tape:
            lr = tf.cast(lr, tf.float32)
            hr = tf.cast(hr, tf.float32)

            sr = self.checkpoint.model(lr, training=True)
            loss_value = self.loss(hr, sr)

        gradients = tape.gradient(loss_value, self.checkpoint.model.trainable_variables)
        self.checkpoint.optimizer.apply_gradients(zip(gradients, self.checkpoint.model.trainable_variables))

        return loss_value

    def evaluate_psnr(self, dataset):
        return utils.evaluate_psnr(self.checkpoint.model, dataset)

    def evaluate_mae(self, dataset):
        return utils.evaluate_mae(self.checkpoint.model, dataset)

    def restore(self):
        if self.checkpoint_manager.latest_checkpoint:
            self.checkpoint.restore(self.checkpoint_manager.latest_checkpoint)
            print(f'Model restored from checkpoint at step {self.checkpoint.step.numpy()}.')


class EdsrTrainer(Trainer):
    def __init__(self,
                 model,
                 checkpoint_dir,
                 learning_rate=PiecewiseConstantDecay(boundaries=[200000], values=[1e-4, 5e-5])):
        super().__init__(model, loss=MeanAbsoluteError(), learning_rate=learning_rate, checkpoint_dir=checkpoint_dir)

    def train(self, train_dataset, valid_dataset, steps=300000, evaluate_every=1000, save_best_only=True):
        super().train(train_dataset, valid_dataset, steps, evaluate_every, save_best_only)


class SrganTrainer:
    def __init__(self,
                 generator,
                 discriminator,
                 adv_loss_weight,
                 label_smoothing=True,
                 noisy_label_prob=0.05,
                 content_loss='VGG19',
                 checkpoint_dir='/work/logs/srganx4',
                 tf_board_dir='/work/logs/srganx4_train/gradient_tape/',
                 learning_rate=PiecewiseConstantDecay(boundaries=[100000], values=[1e-4, 1e-5])):

        if content_loss == 'VGG19':
            self.vgg = models.vgg_19()
        elif content_loss == 'VGG54':
            self.vgg = models.vgg_54()
        else:
            raise ValueError("content_loss must be either 'VGG22' or 'VGG54 for training srgan'")
        
        self.content_loss = content_loss
        self.generator = generator
        self.adv_loss_weight = adv_loss_weight
        self.discriminator = discriminator
        self.generator_optimizer = Adam(learning_rate=learning_rate)
        self.discriminator_optimizer = Adam(learning_rate=learning_rate)

        self.binary_cross_entropy = BinaryCrossentropy(from_logits=False)
        self.mean_squared_error = MeanSquaredError()

        os.makedirs(checkpoint_dir, exist_ok=True)
        self.checkpoint = tf.train.Checkpoint(generator_optimizer=self.generator_optimizer,
                                 discriminator_optimizer=self.discriminator_optimizer,
                                 generator=self.generator,
                                 discriminator=self.discriminator,
                                 step=tf.Variable(0),
                                 psnr=tf.Variable(-1.0))
        
        self.checkpoint_manager = tf.train.CheckpointManager(checkpoint=self.checkpoint,
                                                             directory=checkpoint_dir,
                                                             max_to_keep=3)
        self.log_var = []
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        # tensroboard log dir setup
        self.train_log_dir = tf_board_dir + current_time + '/train'
        self.train_writer = tf.summary.create_file_writer(self.train_log_dir)

        self.label_smoothing = label_smoothing
        self.noisy_label_prob = noisy_label_prob

    def get_log(self):
        return self.log_var

    def train(self, train_dataset, valid_dataset, steps, test_img_path, scale=4, evaluate_every=1000, save_best_only=False):
        gls_metric = Mean()
        dls_metric = Mean()
        dls_hr_metric = Mean()
        dls_sr_metric = Mean()
        con_metric = Mean()
        adv_metric = Mean()
        loss_metrics = {'gen loss': gls_metric, 'disc loss': dls_metric, 'disc loss generated': dls_sr_metric,
                  'disc loss real': dls_hr_metric, 'cont loss': con_metric, 'adv loss': adv_metric}
        
        ckpt_mgr = self.checkpoint_manager
        ckpt = self.checkpoint
        self.now = time.perf_counter()

        restored_steps = ckpt.step.numpy()
        restored_epoch = restored_steps // len(train_dataset)
        total_epochs = steps // len(train_dataset)
        new_step_start = restored_steps % len(train_dataset)
        self.log_var.append('step,train_con_loss,train_adv_loss,train_gen_loss,train_disc_loss,duration')
        
        for epoch in range(restored_epoch, total_epochs):
            for _, data in enumerate(train_dataset):
                step = ckpt.step.numpy()
                ckpt.step.assign_add(1)
                if step < new_step_start:
                    continue

                lr, hr = data[0], data[1]
                con_loss, adv_loss, gen_loss, disc_hr_loss, disc_sr_loss, disc_loss, gradients_of_generator, gradients_of_discriminator \
                = self.train_step(lr, hr)
                # store the losses tmporarily to update the loss metrics
                losses_tmp = {'gen loss': gen_loss, 'disc loss': disc_loss, 'disc loss generated': disc_sr_loss,
                  'disc loss real': disc_hr_loss, 'cont loss': con_loss, 'adv loss': adv_loss}
        
                with self.train_writer.as_default():
                    for var, g in zip(self.generator.trainable_variables, gradients_of_generator):
                        # print(f'{var.name}, shape: {g.shape}')
                        tf.summary.histogram('gen_' + var.name, data=g, step=step)

                    for var, g in zip(self.discriminator.trainable_variables, gradients_of_discriminator):
                        tf.summary.histogram('disc_' + var.name, data=g, step=step)

                    D_norm = self.find_norm(gradients_of_discriminator)
                    G_norm = self.find_norm(gradients_of_generator)
                    tf.summary.scalar('discriminator grad norm', D_norm.numpy(), step=step)
                    tf.summary.scalar('generator grad norm', G_norm.numpy(), step=step)

                # update each of the loss metric using the loss stored in losses_tmp
                for name, metric_obj in loss_metrics.items():
                    metric_obj.update_state(losses_tmp[name])    

                # write all losses at each step to tensorboard 
                utils.write_losses(self.train_writer, loss_metrics, step)

                if step % evaluate_every == 0:
                    duration = time.perf_counter() - self.now
                    psnr_value = self.evaluate_psnr(valid_dataset)
                    disc_acc_generated = self.evaluate_disc(valid_dataset, generated=True)
                    disc_acc_real = self.evaluate_disc(valid_dataset, generated=False)
                    eval_metrics = {'val PSNR':psnr_value.numpy(), 'val disc acc generated':disc_acc_generated.numpy(), 'val disc acc real':disc_acc_real.numpy()}
                   
                    template = 'step {}/{}, train_con_loss {:.4f}, train_adv_loss {:.4f}, train_gen_loss {:.4f}, train_disc_loss {:.4f}, Duration {:.2f}s'
                    to_log = template.format(step, epoch,
                                          loss_metrics['cont loss'].result(),
                                          loss_metrics['adv loss'].result(),
                                          loss_metrics['gen loss'].result(),
                                          loss_metrics['disc loss'].result(),
                                          duration)
                    print(to_log)
                    self.log_var.append(to_log)
                    # write the evaluation metrcis to tensorboard
                    utils.write_eval_metrics(self.train_writer, eval_metrics, epoch)
                    
                    # reset_states for all loss metrics 
                    for _, metric_obj in loss_metrics.items():
                        metric_obj.reset_states()
                    
                    if save_best_only and psnr_value <= ckpt.psnr:
                        self.now = time.perf_counter()
                        # skip saving checkpoint, no PSNR improvement
                        continue

                    self.now = time.perf_counter()
                    ckpt_mgr.save()
                    ckpt.psnr = psnr_value
                    # test generate images and save to file at the end of every epoch (evaluation)
                    self.generate_and_save(test_img_path, epoch, scale, dst_dir=self.train_log_dir)

    def restore(self):
        if self.checkpoint_manager.latest_checkpoint:
            self.checkpoint.restore(self.checkpoint_manager.latest_checkpoint)
            print(f'Model restored from checkpointat step {self.checkpoint.step.numpy()}.')

    @tf.function
    def train_step(self, lr, hr):
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            lr = tf.cast(lr, tf.float32)
            hr = tf.cast(hr, tf.float32)
            sr = self.generator(lr, training=True)
            hr_output = self.discriminator(hr, training=True)
            sr_output = self.discriminator(sr, training=True)

            con_loss = self._content_loss(hr, sr)
            adv_loss = self._adversarial_loss(sr_output)
            gen_loss = con_loss + self.adv_loss_weight * adv_loss
            hr_loss, sr_loss, disc_loss = self._discriminator_loss(hr_output, sr_output)

        gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)
        self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))
        
        # update G twice everytime D is updated to make sure disc_loss does not go to zero 
        with tf.GradientTape() as gen_tape:
            lr = tf.cast(lr, tf.float32)
            hr = tf.cast(hr, tf.float32)
            sr = self.generator(lr, training=True)
            sr_output = self.discriminator(sr, training=True)

            con_loss = self._content_loss(hr, sr)
            adv_loss = self._adversarial_loss(sr_output)
            gen_loss = con_loss + self.adv_loss_weight * adv_loss  
        gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))

        return con_loss, adv_loss, gen_loss, hr_loss, sr_loss, disc_loss, gradients_of_generator, gradients_of_discriminator

    @tf.function
    def _content_loss(self, hr, sr):
        sr = preprocess_input(sr)
        hr = preprocess_input(hr)
        sr_features = self.vgg(sr) / 12.75
        hr_features = self.vgg(hr) / 12.75
        return self.mean_squared_error(hr_features, sr_features)

    def _adversarial_loss(self, sr_out):
        return self.binary_cross_entropy(tf.ones_like(sr_out), sr_out)

    def _discriminator_loss(self, hr_out, sr_out):
        if not self.label_smoothing:
            hr_loss = self.binary_cross_entropy(tf.ones_like(hr_out), hr_out)
            sr_loss = self.binary_cross_entropy(tf.zeros_like(sr_out), sr_out)
        else:
            noisy_pos_label = self.noisy_label(tf.ones_like(hr_out), self.noisy_label_prob)
            noisy_neg_label = self.noisy_label(tf.zeros_like(sr_out), self.noisy_label_prob)
            pos_label_smooth = self.smooth_pos_labels(noisy_pos_label)
            neg_label_smooth = self.smooth_neg_labels(noisy_neg_label)
            sr_loss = self.binary_cross_entropy(neg_label_smooth, sr_out)
            hr_loss = self.binary_cross_entropy(pos_label_smooth, hr_out)
        return hr_loss, sr_loss, hr_loss + sr_loss
    
    def generate_and_save(self, test_img_path, epoch, scale, dst_dir):
        utils.generate_and_save(self.checkpoint.generator, test_img_path, epoch,
                                scale, patch_size=50, dst_dir=dst_dir)

    def evaluate_psnr(self, dataset):
        return utils.evaluate_psnr(self.checkpoint.generator, dataset)
    
    def evaluate_disc(self, dataset, generated):
        return utils.evaluate_disc(self.checkpoint.generator, self.checkpoint.discriminator,
                                   dataset, generated=generated)

    def smooth_pos_labels(self, y):
        # assign a random integer in range [0.7, 1.2] for positive class
        # return tf.subtract(y, tf.fill(y.shape, 0.3)) + (np.random.random(y.shape) * 0.5)
        return y - 0.3 + (np.random.random(y.shape) * 0.5)
        
    def smooth_neg_labels(self, y):
        # and [0.0, 0.3] for negative class
        return y + np.random.random(y.shape) * 0.3
    
    def noisy_label(self, y, prob):
        # determine the number of labels to flip
        num_label = int(prob * y.shape[0])
        # choose labels to flip
        flip_idx = np.random.choice([i for i in range(y.shape[0])], size=num_label)

        op_list = []
        for i in range(int(y.shape[0])):
            if i in flip_idx:
                op_list.append(tf.subtract(1., y[i]))
            else:
                op_list.append(y[i])    
        output = tf.stack(op_list)

        return output

    def find_norm(self, grad):
        norm = tf.math.sqrt(sum([tf.math.reduce_sum(tf.math.square(g)) for g in grad]))
        return norm 