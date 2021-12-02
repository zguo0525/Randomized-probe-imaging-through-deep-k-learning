# tensorflow script used in "Randomized probe imaging through deep k-learning"
# written and maintained by Abe Levitan and Zhen Guo
# =============================================================================
"""Contains script to train E2E architecture for simulation data."""

import tensorflow as tf
from networks import *
from datetime import date
from scipy import io
import numpy as np
import time

prefix = '../data/' + 'Simulated/R_Sweep/'

if __name__ == "__main__":
    
    from tqdm import tqdm
    
    # loop through Rs
    photon_levels = 1e4

    Rs = [0.25, 0.5, 1, 2]

    for R in [Rs]:

        expanded_probe = np.load(prefix + 'probe-flatten-R-%0.2f.npy' % R).astype(np.float32)
        expanded_probe = tf.convert_to_tensor(expanded_probe)

        chn1 = expanded_probe.shape[-1]

        data_input = np.load(prefix + 'tr_patterns-flatten-R-%0.2f-phperpix-%0.2f.npy' %(R, photon_level)).astype(np.float32)
        data_input = norm_to_255(data_input)

        chn2 = data_input.shape[-1]

        data_output = np.load(prefix + 'tr_images-R-%0.2f.npy' % R)
        data_output = np.angle(data_output)
        data_output = tf.expand_dims(data_output, axis=-1)

        efficient_unet = get_efficient_unet_b7((256, 256, chn1+chn2), 
                                  out_channels=1,  
                                  dropout=0.25, 
                                  pretrained=False, 
                                  concat_input=False)

        def model():
            inputs = tf.keras.layers.Input(shape=(256, 256, chn2))
            padded_in_probe = tf.expand_dims(expanded_probe,axis=0)
            padded_in_probe = tf.tile(padded_in_probe, multiples=[tf.shape(inputs)[0],1,1,1])
            padded_in = tf.concat([inputs, padded_in_probe], axis=-1)

            output = efficient_unet(padded_in)

            model = tf.keras.Model(inputs=inputs, outputs=output)
            return model

        model = model()
        model.summary()

        batch = 10
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

        history = model.fit(data_input, data_output, 
                  epochs=200, 
                  batch_size=batch, 
                  validation_split=0.1, 
                  verbose = 1, 
                  callbacks=callback_list, 
                  shuffle = True)
        
        model.save(prefix + 'E2E-R-' + str(R) + '-photon-' + str(photon_level) + '.h5')
        
        test_input = np.load(prefix + 'test_patterns-flatten-R-%0.2f-phperpix-%0.2f.npy' %(R, photon_level)).astype(np.float32)
        test_input = norm_to_255(test_input)
        test_output = model.predict(test_input, batch_size=batch)
        
        #used to calculate the scale factor
        from scipy import stats
        # get the training output
        rec_training_output = model.predict(data_input, batch_size=10, verbose=1)
        # reshape the training output
        rec_training_output = rec_training_output.reshape(1,-1)
        # reshape the data output
        data_output = np.array(data_output)
        data_output = data_output.reshape(1,-1)
        # calculate the scale and shift factor
        slope, intercept, r_value, p_value, std_err = stats.linregress(rec_training_output, data_output)
        print('slope, intercept, r_value, p_value, std_err', slope, intercept, r_value, p_value, std_err)

        #rescale the test output
        test_output = np.array(test_output)
        test_output = test_output.reshape(1,-1)
        test_output = test_output * slope + intercept
        test_output = test_output.reshape((100, 256, 256))
        

        np.save('End-to-End-test-output-R-' + str(R) + '-photon-' + str(photon_level), test_output)
