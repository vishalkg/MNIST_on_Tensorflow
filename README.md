MNIST_on_Tensorflow

Data is taken from kaggle having 32000 samples.
*Training Size: 30000
*Testing Size: 2000
*Accuracy: 99.3%

Network Configuration | Filter Size | #Filters | Max Pooling/ReLU
----------------------|-------------|----------|-----------------
Convolutional Layer 1 | 5x5 | 16 | Yes            
Convolutional Layer 2 | 5x5 | 32 | No            
Convolutional Layer 3 | 2x2 | 64 | Yes
Convolutional Layer 4 | 2x2 | 128 | No
Fully Connected Layer 1 | N/A | 512 | Yes
Fully Connected Layer 2 | N/A | 256 | Yes
Fully Connected Layer 3 | N/A | 10 | No (Logistic Layer)

Hardware | NVIDIA GeForce 670 Ti OEM
Running Time | 6-7 Minutes
