# tensorflow script used in "Randomized probe imaging through deep k-learning"
# written and maintained by Abe Levitan and Zhen Guo
# =============================================================================
"""Contains script to train deep k-learning architecture (generative) for simulation data."""

import tensorflow as tf
from networks import *
from datetime import date
from scipy import io
import numpy as np
import time


# we load the EfficientnetB0 for computing the representation loss
feature_model = tf.keras.applications.EfficientNetB0(include_top=False, weights='imagenet', input_tensor=None,
                                                        input_shape=(256, 256, 3), pooling=None, classes=1000,
                                                        classifier_activation='softmax')
# we use MAE for the representation loss
mae = tf.keras.losses.MeanAbsoluteError()


def generator_loss(disc_generated_output, gen_output, target):
    """define geneartor loss
    """
    generative_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)(tf.ones_like(disc_generated_output), disc_generated_output)

    supervised_loss = npcc(target, gen_output)

    targeted = feature_model(tf.concat([target, target, target], -1), training=False)
    generated = feature_model(tf.concat([gen_output, gen_output, gen_output], -1), training=False)

    representation_loss = mae(targeted, generated)

    total_gen_loss = alpha * generative_loss + 1/8 * representation_loss + supervised_loss 

    return total_gen_loss, generative_loss, supervised_loss


@tf.function
def train_step(input_image, target, epoch):
    """overwrite the training step to enable generative training
    """
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        # get the output of the generator
        gen_output = generator(input_image, training=True)
        
        # get one of the channel in the input image for the discriminator
        disc_real_output = discriminator([input_image[:, :, :, :1], target], training=True)
        disc_generated_output = discriminator([input_image[:, :, :, :1], gen_output], training=True)
        
        # calculate the loss for the generator
        total_gen_loss, generative_loss, supervised_loss = generator_loss(disc_generated_output, gen_output, target)
        # calculate the loss for the discriminator
        disc_loss = discriminator_loss(disc_real_output, disc_generated_output)

    generator_gradients = gen_tape.gradient(total_gen_loss,
                                          generator.trainable_variables)
    discriminator_gradients = disc_tape.gradient(disc_loss,
                                               discriminator.trainable_variables)

    # update the weights in generator and discriminator
    generator_optimizer.apply_gradients(zip(generator_gradients,
                                          generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(discriminator_gradients,
                                              discriminator.trainable_variables))

    return total_gen_loss, generative_loss, supervised_loss, disc_loss

def Callback_EarlyStopping(LossList, min_delta=0.02, patience=20):
    """early stop function for custom training
    """
    # No early stopping for 2*patience epochs 
    if len(LossList)//patience < 2 :
        return False
    # Mean loss for last patience epochs and second-last patience epochs
    mean_previous = np.mean(LossList[::-1][patience:2*patience]) #second-last
    mean_recent = np.mean(LossList[::-1][:patience]) #last
    # you can use relative or absolute change
    delta_abs = np.abs(mean_recent - mean_previous) #abs change
    delta_abs = np.abs(delta_abs / mean_previous)  # relative change
    if delta_abs < min_delta :
        print("*CB_ES* Loss didn't change much from last %d epochs"%(patience))
        print("*CB_ES* Percent change in loss value:", delta_abs*1e2)
        return True
    else:
        return False

def train(dataset, valid_dataset, epochs):
    """define train function
    """
    gen_loss_list = []
    pure_gen_loss = []
    supervised_loss_list = []
    disc_loss_list = []
    val_loss_list = []

    for epoch in range(epochs):
        start = time.time()

        for recon_batch, image_batch in dataset:
            gen_total_loss, gen_gan_loss, supervised_loss, disc_loss = train_step(recon_batch, image_batch, epoch)
            supervised_loss_list.append(supervised_loss)
            gen_loss_list.append(gen_total_loss)
            disc_loss_list.append(disc_loss)
            pure_gen_loss.append(gen_gan_loss)

        val_temp = []
        for recon_batch, image_batch in valid_dataset:
            decoded = generator.predict(recon_batch, batch_size=batch,verbose=0)
            # we used npcc loss to monitor the reconstruction
            val_loss = npcc(image_batch, decoded)
            # this add the npcc loss by one to make the loss positive, therefore,
            # the callback function later on 
            val_loss += 1
            val_temp.append(val_loss)
        val_loss_list.append(np.mean(val_temp))

        reducelr = Callback_EarlyStopping(val_loss_list, min_delta=0.001, patience=10)
        if reducelr:
            print("Callback_reduce lr signal received at epoch= %d/%d"%(epoch,epochs))
            # reduce lr by half when validation loss plateau
            old_lr = generator_optimizer.lr.read_value()
            generator_optimizer.lr.assign(0.5 * old_lr)
            discriminator_optimizer.lr.assign(0.5 * old_lr)

        stopEarly = Callback_EarlyStopping(val_loss_list, min_delta=0.001, patience=20)
        if stopEarly:
            print("Callback_EarlyStopping signal received at epoch= %d/%d"%(epoch,epochs))
            print("Terminating training ")
            break

        print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))
        print ('supervised_loss = ' + str(supervised_loss.numpy()))
        print ('val_loss = ' + str(np.mean(val_temp)))
        print ('gen_loss = ' + str(gen_total_loss.numpy()))
        print ('disc_loss = ' + str(disc_loss.numpy()))
        print ('pure_gen_loss = ' + str(gen_gan_loss.numpy()))
        
    return gen_loss_list, supervised_loss_list, val_loss_list, disc_loss_list

pretrain = True
prefix = '../data/' + 'Simulated/Fixed_R_Noise_Sweep/'
R = 0.5

if __name__ == "__main__":
    
    date_string = date.today().strftime('%Y-%m-%d')
    alphas = [1, 1/2, 1/4, 1/8, 1/16, 1/32]
    
    # loop through photon levels
    photon_levels = [1, 10, 100, 1e3]
    for photon_level in photon_levels:
        # loop through alpha values 
        for alpha in alphas:
            
            # some training parameters 
            dropout = 0.25
            batch = 10
            epoch = 200
            validation_split = 0.1

            generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
            discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

            # get generator
            generator = get_efficient_unet_b7((256, 256, 3), 
                                        out_channels=1,  
                                        dropout=dropout, 
                                        pretrained=pretrain, 
                                        concat_input=False)
            generator.summary()

            # get discriminator
            discriminator = Discriminator()
            discriminator.summary()

            # load and normalize the training data
            data_input = np.load(prefix + 'tr-reconstruction-R-%0.2f-phperpix-%d-iters-1-lr-1.00.npy' % (R, photon_level))
            data_input = np.expand_dims(data_input, axis=-1)
            data_input = norm_to_255(data_input)
            data_input = np.concatenate((data_input, data_input, data_input), axis=-1)

            # load the training ground truth data
            data_output = np.load(prefix + 'tr_images.npy')
            data_output = np.angle(data_output)
            data_output = np.expand_dims(data_output, axis=-1)

            # load and normalize the testing data
            test_input = np.load(prefix + 'test-reconstruction-R-%0.2f-phperpix-%d-iters-1-lr-1.00.npy' % (R, photon_level))
            test_input = np.expand_dims(test_input, axis=-1)
            test_input = norm_to_255(test_input)
            test_input = np.concatenate((test_input, test_input, test_input), axis=-1)

            # getting training set and validation set from the training dataset
            recon_dataset = tf.data.Dataset.from_tensor_slices(data_input[:int(data_input.shape[0]*(1-validation_split))])
            images_dataset = tf.data.Dataset.from_tensor_slices(data_output[:int(data_input.shape[0]*(1-validation_split))])

            valid_recon = tf.data.Dataset.from_tensor_slices(data_input[int(data_input.shape[0]*(1-validation_split)):])
            valid_images = tf.data.Dataset.from_tensor_slices(data_output[int(data_input.shape[0]*(1-validation_split)):])

            train_dataset = tf.data.Dataset.zip((recon_dataset, 
                                                images_dataset)).shuffle(buffer_size=1024, reshuffle_each_iteration=True).batch(batch)

            valid_dataset = tf.data.Dataset.zip((valid_recon, valid_images)).batch(batch*2)

            gen_loss_list0, supervised_loss_list0, val_loss_list0, disc_loss_list0 = train(train_dataset, valid_dataset, epoch)

            test_output0 = generator.predict(test_input,
                         batch_size=batch,
                         verbose=1)

            if pretrain is True:
                generator.trainable = True
                generator.summary()

                generator_optimizer = tf.keras.optimizers.Adam(1e-4, beta_1=0.5)
                discriminator_optimizer = tf.keras.optimizers.Adam(1e-4, beta_1=0.5)

                gen_loss_list, supervised_loss_list, val_loss_list, disc_loss_list = train(train_dataset, valid_dataset, epoch)

                test_output = generator.predict(test_input,
                             batch_size=batch,
                             verbose=1)
                #save the model
                generator.save(prefix + 'generative-pretrained-R-' + str(R) + '-phperpix-' + str(photon_level) + '-alpha-' + str(alpha) + '.h5')

            else:
                gen_loss_list = gen_loss_list0
                val_loss_list = val_loss_list0
                supervised_loss_list = supervised_loss_list0
                disc_loss_list = disc_loss_list0
                test_output = test_output0

                generator.save(prefix + 'generative-not-pretrained-R-' + str(R) + '-phperpix-' + str(photon_level) + '-alpha-' + str(alpha) + '.h5')

            #used to calculate the scale factor
            from scipy import stats
            # get the training output
            rec_training_output = generator.predict(data_input, batch_size=10, verbose=1)
            # reshape the training output
            rec_training_output = rec_training_output.reshape(1,-1)
            # reshape the data output
            data_output = data_output.reshape(1,-1)
            # calculate the scale and shift factor
            slope, intercept, r_value, p_value, std_err = stats.linregress(rec_training_output, data_output)
            print('slope, intercept, r_value, p_value, std_err', slope, intercept, r_value, p_value, std_err)

            #rescale the test output
            test_output = test_output.reshape(1,-1)
            test_output = test_output * slope + intercept
            test_output = test_output.reshape((100, 256, 256))
                
            # save data
            if pretrain is True:
                
                io.savemat(date_string + '-pretrained-alpha-%0.2f-R-%0.2f-photon-%0.2f.mat' % (alpha, R, photon_level),
                           {'gen_loss_list_round1': np.array(gen_loss_list0),
                            'supervised_loss_round1': np.array(supervised_loss_list0),
                            'val_loss_list_round1': np.array(val_loss_list0),
                            'disc_loss_list_round1': np.array(disc_loss_list0),
                            'rec_test_output_round1': np.array(test_output0),
                            'gen_loss_list': np.array(gen_loss_list),
                            'supervised_loss_list': np.array(supervised_loss_list),
                            'val_loss_list': np.array(val_loss_list),
                            'disc_loss_list': np.array(disc_loss_list),
                            'rec_test_output': np.array(test_output)
                           })
            else:
                
                io.savemat(date_string + '-not-pretrained-alpha-%0.2f-R-%0.2f-photon-%0.2f.mat' % (alpha, R, photon_level),
                           {'gen_loss_list': np.array(gen_loss_list),
                            'supervised_loss_list': np.array(supervised_loss_list),
                            'val_loss_list': np.array(val_loss_list),
                            'disc_loss_list': np.array(disc_loss_list),
                            'rec_test_output': np.array(test_output)
                           })
