#**Traffic Sign Recognition** 

##Writeup

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is  = 34799
* The size of the validation set is = 4410
* The size of test set is = 12630
* The shape of a traffic sign image is = (32, 32, 3)
* The number of unique classes/labels in the data set is = 43

####2. Include an exploratory visualization of the dataset.

Kindly check the visualization in the notebook or the HTML ...

###Design and Test a Model Architecture

####1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because to work with only 1 channel

Then I normalized the data to have mean zero and give it the same variance

I decided to generate additional data but it didn't improved my accuracy, only get the process slowed, I let the augement code commented, probably I need to work with a different model to use the rotation, but the end result was good for this project so I decide to leave like that for now, but for sure increase the number o variations will increase the accuracy.

####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 3x3     	| 1x1 stride, same padding, outputs 32x32x64 	|
| RELU					|												|
| Convolution 3x3       | 1x1 stride, same padding, outputs 32x32x64    |
| RELU                  |                                               |
| Max pooling	      	| 2x2 stride,  outputs 16x16x64 				|
| Convolution 3x3       | 1x1 stride, same padding, outputs 16x16x128   |
| RELU                  |                                               |
| Convolution 3x3       | 1x1 stride, same padding, outputs 16x16x128   |
| RELU                  |                                               |
| Max pooling           | 2x2 stride,  outputs 8x8x128                  |
| Fully connected       | Input: 8x8x128, flattens and outputs 8192     |
| Dropout               | Probability: 0.5                              |
| Fully connected       | Input: 8192, outputs 43                       |
| Dropout               | Probability: 0.5                              |
| Softmax               |                                               |

 


####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an ....

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

* What was the first architecture that was tried and why was it chosen?
My solution started with LeNet , done in the past videos, I tried to implemente AlexNet, but without success, I return to LeNet to finish the project in time. I will try to use AlexNet as soon as I've the time to do it.
* What were some problems with the initial architecture?
LeNet is not deep and wide enough to do this kind of job, it was created to character recognition, it's very limited compared to other new approaches.
* How was the architecture adjusted and why was it adjusted?
I tried to used AlexNet to create more deep, but I didn't fill confident about where I was going, so I return to Lenet to finish the job in time, since the number of classes were small.
* Which parameters were tuned? How were they adjusted and why?
almost all of them, tuned the learning, rate, epochs, number of layers, size of it... runned at least 30x to find the current results With different numbers. I know that it can be automated with some kind of gridsearch, but by now try and error make more clear to understand where I was going.
* What are some of the important design choices and why were they chosen?
I think the the choise of the optimizer make a real diference here, Adam improved a lot the performance from pure SGD bringing my final model results to:
0.983 for Validation
0.979 for Testing

with the final parameters
EPOCHS = 100 BATCH_SIZE = 128 mu = 0 sigma = 0.1 rate = 0.0001 Optimizer = ADAM

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

<img src="/traffic-signs-data/test-images/0.jpg" width="200" height="200">
<img src="/traffic-signs-data/test-images/1.jpg" width="200" height="200">
<img src="/traffic-signs-data/test-images/2.jpg" width="200" height="200">
<img src="/traffic-signs-data/test-images/3.jpg" width="200" height="200">
<img src="/traffic-signs-data/test-images/4.jpg" width="200" height="200">

I choosed low resolution images with noise so it would be more difficult to classify but the results are 100% right...! I even try different images, and still 100% so I was pretty convinced that the model works for now. The softmax for the top 5 are included below.

The comparison against the Test and the Real Test(only 5 images) are: 98% for the Test and 100% for the Real Test.

* 33 ['Turn right ahead']
[ 1.  0.  0.  0.  0.] ['Turn right ahead', 'Speed limit (20km/h)', 'Speed limit (30km/h)', 'Speed limit (50km/h)', 'Speed limit (60km/h)']
* 17 ['No entry']
[ 1.  0.  0.  0.  0.] ['No entry', 'Speed limit (20km/h)', 'Speed limit (30km/h)', 'Speed limit (50km/h)', 'Speed limit (60km/h)']
* 5 ['Speed limit (80km/h)']
[ 1.  0.  0.  0.  0.] ['Speed limit (80km/h)', 'Speed limit (20km/h)', 'Speed limit (30km/h)', 'Speed limit (50km/h)', 'Speed limit (60km/h)']
* 14 ['Stop']
[ 1.  0.  0.  0.  0.] ['Stop', 'Speed limit (20km/h)', 'Speed limit (30km/h)', 'Speed limit (50km/h)', 'Speed limit (60km/h)']
* 35 ['Ahead only']
[ 1.  0.  0.  0.  0.] ['Ahead only', 'Speed limit (20km/h)', 'Speed limit (30km/h)', 'Speed limit (50km/h)', 'Speed limit (60km/h)']


