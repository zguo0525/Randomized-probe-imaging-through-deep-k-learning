# tensorflow script used in "Randomized probe imaging through deep k-learning"
# written and maintained by Abe Levitan and Zhen Guo
# =============================================================================
"""Contains script to train deep k-learning architecture."""

import tensorflow as tf
from networks import *
from datetime import date
from scipy import io
import numpy as np
import time

pretrain = True
prefix = '../data/' + 'Simulated/R_Sweep/'
photon_level = 1e4

if __name__ == "__main__":

    date_string = date.today().strftime('%Y-%m-%d')
    
    # loop through R values
    for R in [0.25, 0.5, 1, 2]:

        dropout = 0.25
        epoch = 200
        batch = 10

        # define model
        model = get_efficient_unet_b7((256, 256, 3), 
                                    out_channels=1,  
                                    dropout=dropout, 
                                    pretrained=pretrain, 
                                    concat_input=False)

        model.summary()

        # load and normalize the training data
        data_input = np.load(prefix + 'tr-reconstruction-R-%0.2f-phperpix-%d-iters-1-lr-1.00.npy' % (R, photon_level))
        data_input = np.expand_dims(data_input, axis=-1)
        data_input = norm_to_255(data_input)
        data_input = np.concatenate((data_input, data_input, data_input), axis=-1)

        # load the training ground truth data
        data_output = np.load(prefix + 'tr_images-R-%0.2f.npy' % R)
        data_output = np.angle(data_output)
        data_output = np.expand_dims(data_output, axis=-1)

        # load and normalize the testing data
        test_input = np.load(prefix + 'test-reconstruction-R-%0.2f-phperpix-%d-iters-1-lr-1.00.npy' % (R, photon_level))
        test_input = np.expand_dims(test_input, axis=-1)
        test_input = norm_to_255(test_input)
        test_input = np.concatenate((test_input, test_input, test_input), axis=-1)

        optadam = tf.keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False)

        reducelr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', 
                                                        factor=0.5, 
                                                        patience=10, 
                                                        verbose=1, 
                                                        mode='auto', 
                                                        min_delta=0.001,
                                                        cooldown=0, 
                                                        min_lr=1e-8)

        earlystop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20)

        callback_list = [earlystop,
                        reducelr]

        model.compile(optimizer=optadam, loss=npcc)

        #train model
        history1 = model.fit(data_input, data_output, 
                epochs=epoch, 
                batch_size=batch, 
                validation_split=0.1, 
                verbose = 1, 
                callbacks=callback_list, 
                shuffle = True)

        train_loss1 = history1.history['loss']
        validation_loss1 = history1.history['val_loss']

        test_output0 = model.predict(test_input,
                    batch_size=batch,
                    verbose=1)

        if pretrain is True:
            model.trainable = True
            model.summary()

            optadam = tf.keras.optimizers.Adam(learning_rate=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False)
            model.compile(optimizer=optadam, loss=npcc)

            reducelr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', 
                                                            factor=0.5, 
                                                            patience=5, 
                                                            verbose=1, 
                                                            mode='auto', 
                                                            min_delta=0.001,
                                                            cooldown=0, 
                                                            min_lr=1e-9)

            earlystop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

            callback_list = [earlystop,
                            reducelr]

            history2 = model.fit(data_input, data_output, 
                    epochs=epoch, 
                    batch_size=batch, 
                    validation_split=0.1, 
                    verbose = 1, 
                    callbacks=callback_list, 
                    shuffle = True)

            train_loss2 = history2.history['loss']
            validation_loss2 = history2.history['val_loss']

        test_output = model.predict(test_input,
                        batch_size=batch,
                        verbose=1)

        #used to calculate the scale factor
        from scipy import stats
        # get the training output
        rec_training_output = model.predict(data_input, batch_size=10, verbose=1)
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
            
            #save model
            model.save(prefix + 'pretrained-R-' + str(R) + '-peak-' + str(photon_level) + '.h5')

            io.savemat(date_string + + 'pretrained-R-' + str(R) + '-peak-' + str(photon_level) + '.mat',{'train_loss1':np.array(train_loss1),
                                                'validation_loss1':np.array(validation_loss1),
                                                'train_loss2':np.array(train_loss2),
                                                'validation_loss2':np.array(validation_loss2),
                                                'rec_test_output0':np.array(test_output0),
                                                'rec_test_output':np.array(test_output)})
        else:
            model.save(prefix + 'not-pretrained-R-' + str(R) + '-peak-' + str(photon_level) + '.h5')

            io.savemat(date_string + 'not-pretrained-R-' + str(R) + '-peak-' + str(photon_level) + '.mat',{'train_loss':np.array(train_loss1),
                                            'validation_loss':np.array(validation_loss1),
                                            'rec_test_output':np.array(test_output0)})
