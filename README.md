Project: Behavioral Cloning
====================
#Augmentation
8000+ lines in Udacity data set is not enough for a full train, especially for recovery and for generalization for track 2. I applied following augmentation technics:   
1. Select camera from (left, center, right)  
2. Randomly change brightness  
3. Transition horizontally  
4. Crop, to reduce non-valued information  
5. Random shadow, for the second track  
6. Flip  

In additional, I tried to draw on images previous value of steering (in fact - current direction),  but without success.  

#Model
                     
lambda_1 (Lambda)                (None, 103, 320, 3)   0           lambda_input_1[0][0]             

convolution2d_1 (Convolution2D)  (None, 52, 160, 3)    228         lambda_1[0][0]                   

elu_1 (ELU)                      (None, 52, 160, 3)    0           convolution2d_1[0][0]            

convolution2d_2 (Convolution2D)  (None, 26, 80, 24)    1824        elu_1[0][0]                      

elu_2 (ELU)                      (None, 26, 80, 24)    0           convolution2d_2[0][0]            

convolution2d_3 (Convolution2D)  (None, 13, 40, 36)    21636       elu_2[0][0]                      

elu_3 (ELU)                      (None, 13, 40, 36)    0           convolution2d_3[0][0]            

convolution2d_4 (Convolution2D)  (None, 7, 20, 48)     15600       elu_3[0][0]                      

elu_4 (ELU)                      (None, 7, 20, 48)     0           convolution2d_4[0][0]            

convolution2d_5 (Convolution2D)  (None, 4, 10, 64)     27712       elu_4[0][0]                      

elu_5 (ELU)                      (None, 4, 10, 64)     0           convolution2d_5[0][0]            

convolution2d_6 (Convolution2D)  (None, 2, 5, 64)      36928       elu_5[0][0]                      

elu_6 (ELU)                      (None, 2, 5, 64)      0           convolution2d_6[0][0]            

flatten_1 (Flatten)              (None, 640)           0           elu_6[0][0]                      

dropout_1 (Dropout)              (None, 640)           0           flatten_1[0][0]                  

dense_1 (Dense)                  (None, 1164)          746124      dropout_1[0][0]                  

dropout_2 (Dropout)              (None, 1164)          0           dense_1[0][0]                    

elu_7 (ELU)                      (None, 1164)          0           dropout_2[0][0]                  

dense_2 (Dense)                  (None, 100)           116500      elu_7[0][0]                      

dropout_3 (Dropout)              (None, 100)           0           dense_2[0][0]                    

elu_8 (ELU)                      (None, 100)           0           dropout_3[0][0]                  

dense_3 (Dense)                  (None, 50)            5050        elu_8[0][0]                      

dropout_4 (Dropout)              (None, 50)            0           dense_3[0][0]                    

elu_9 (ELU)                      (None, 50)            0           dropout_4[0][0]                  

dense_4 (Dense)                  (None, 10)            510         elu_9[0][0]                      

dropout_5 (Dropout)              (None, 10)            0           dense_4[0][0]                    

elu_10 (ELU)                     (None, 10)            0           dropout_5[0][0]                  

dense_5 (Dense)                  (None, 1)             11          elu_10[0][0]                     

Total params: 972,123

#Hyperparameters

* batch_size = 512
* samples_per_epoch = 39936
* epochs = 10

#Result
Totally I make 40 runs with different model architectures, different augmentations, and different hyperparameters. The final model passed track 1 and half of track 2.  
The problem with track 2 is the sharp turn to the right on the descent after the tunnel: set throttle to 0 is not enough to decrease speed value, I need a brake to pass it. 
