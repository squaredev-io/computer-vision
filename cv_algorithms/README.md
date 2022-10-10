# Computer Vision

## Introduction

Computer vision is a field of artificial intelligence (AI) that enables computers and systems to derive meaningful information from digital images, videos and other visual inputs — and take actions or make recommendations based on that information. If AI enables computers to think, computer vision enables them to see, observe and understand.

Computer vision works much the same as human vision, except humans have a head start. Human sight has the advantage of lifetimes of context to train how to tell objects apart, how far away they are, whether they are moving and whether there is something wrong in an image.

Computer vision trains machines to perform these functions, but it has to do it in much less time with cameras, data and algorithms rather than retinas, optic nerves and a visual cortex. Because a system trained to inspect products or watch a production asset can analyse thousands of products or processes a minute, noticing imperceptible defects or issues, it can quickly surpass human capabilities.

## 1. Computer Vision for Images

### 1.1 What is an image?

Computers work in binary mode, means that all the data are essentially represented as 0 and 1. All data must be converted into binary in order for a computer to process it. Images are no exception as they can be represented as tabular data. To create the picture, a grid can be set out and the squares, known as pixels, coloured (0 - black and 1 - white) - these are the so called black/white images:

![Image representation](cv_algorithms/images/Untitled.png)

#### **1.1.1 Depth**

One of the most important characteristics of images is the depth. Images are not only black and white - as shown above - but they can have all the variations between black and white, forming the so called greyscale images. To achieve this more bits are required for each pixel. The number of bits determines the range of colour. This is known as an image's depth.

For example, using a depth of two, ie two bits per pixel, would allow four possible colours, such as:

- 00 - black
- 01 - dark grey
- 10 - light grey
- 11 - white

![Image with depth](cv_algorithms/images/depth.png)

Each extra bit doubles the range of colours that are available:

- 1 bit per pixel (0 or 1) - two possible colours
- 2 bits per pixel (00 to 11) - four possible colours
- 3 bits per pixel (000 to 111) - eight possible colours
- 4 bits per pixel (0000 to 1111) - 16 possible colours
- 8 bits per pixel (0000 0000 to 1111 1111) - 256 possible colours

The more colours an image requires, the more bits per pixel are needed. Therefore, the more the colour depth, the larger the image file will be.

#### **1.1.2 Size**

Image size is simply the number of pixels that an image contains. It is expressed as height and width. For example:

- 256 × 256
- 640 × 480
- 1024 × 764

#### **1.1.3 Resolution**

Image quality is affected by the resolution of the image. The resolution of an image is a way of describing how tightly packed the pixels are. In a low-resolution image, the pixels are larger and therefore, fewer are needed to fill the space. This results in images that look blocky or pixelated. An image with a high resolution has more pixels, so it looks a lot better when it is enlarged or stretched. The higher the resolution of an image, the larger its file size will be.

#### **1.1.4 Color**

As we already mention, grayscale images are represented as grids of values (depending of the depth the different values could be from 2 to 256). The most common form of coloured images are the RGB. RGB stands for Red, Green and Blue. In order to create those colorful images we need to stack 3 grids, one on top of the other. In this case grids are called channels.  Each pixel is a mix of those three channels.

![Image with colors](cv_algorithms/images/colors.png)

#### **Image representation for Python**

Images are represented in a vector format in python (usually as numpy arrays) . The most common representation is the following:

(Number of images, Height of Image, Width of Image, Channels of Image) = (100, 28, 28, 3)

### 1.2 Data Science Challenges for Images

#### **1.2.1 Image Classification - supervised approach**

Image Classification is a fundamental task that attempts to comprehend an entire image as a whole. The goal is to classify the image by assigning it to a specific label. Typically, Image Classification refers to images in which only one object appears and is analysed. Images are expected to have only one class for each image. Image classification models take an image as input and return a prediction about which class the image belongs to.

Image classification is a supervised learning problem: define a set of target classes (objects to identify in images), and train a model to recognise them using labeled example photos. Early computer vision models relied on raw pixel data as the input to the model.

![Image classification](cv_algorithms/images/catdog.png)

##### **1.2.1.1 Modelling approaches for supervised image classification**

##### *1.2.1.1.1 Convolutional Neural Networks (CNNs)*

The convolutional Neural Network CNN works by getting an image, designating it some weightage based on the different objects of the image, and then distinguishing them from each other. CNN requires very little pre-process data as compared to other deep learning algorithms. One of the main capabilities of CNN is that it applies primitive methods for training its classifiers, which makes it good enough to learn the characteristics of the target object.

CNN is based on analogous architecture, as found in the neurons of the human brain, specifically the Visual Cortex. Each of the neurons gives a response to a certain stimulus in a specific region of the visual area identified as the Receptive field. These collections overlap in order to contain the whole visual area.

CNN algorithm is based on various modules that are structured in a specific workflow that are listed as follows:

- Input Image
- Convolution Layer (Kernel)
- Pooling Layer
- Classification — Fully Connected Layer

![CNN](cv_algorithms/images/cnn.png)

Input Image:

CNN takes an image as an input, distinguishes its objects based on three color planes, and identifies various color spaces. It also measures the image dimensions. In order to explain this process, we will give an example of an RGB image given below.

In this image, we have various colors based on the three-color plane that is Red, Green, and Blue, also known as RGB. The various color spaces are then identified in which images are found, such as RGB, CMYK, Grayscale, and many more. It can become a tedious task while measuring the image dimensions as an example if the image is perse 8k (*7680x4320*). Here comes one of the handy capabilities of CNN that it reduces the image’s dimension to the point that it is easier to process, which also maintaining all of its features in one piece. This is done so that a better prediction is obtained. This ability is critical when designing architectures having not only better learning features but also can work on massive datasets of images.

Convolution Layer (Kernel):

The Kernel of CNN works on the basis of the following formula.

Image Dimensions = n1 x n2 x 1 where n1 = height, n2 = breadth, and 1 = Number of channels such as RGB.

So, as an example, the formula will become I D = 5 x 5 x 1. We will explain this using the image given below.

![kernel](cv_algorithms/images/kernel.png)

In this image, the green section shows the 5 x 5 x 1 formula. The yellow box evolves from the first box till last, performing the convolutional operation on every 3x3 matrix. This operation is called Kernel (K) and work on the basis of the following binary algorithm.

| ----------|    KERNEL (K)      |    ----------       |
|-----------|------|--------|
| 1 | 0 | 1 |
| 0 | 1 | 0 |
| 1 | 0 | 1 |


Based on the above image and kernel(K) the centered yellow  pixel will be the result:

(1x1)+(0x1)+(1x1)+(1x0)+(1x1)+(0x0)+(1x1)+(0x0)+(0x1)=1+0+1+0+1+0+1+0+0=4 

This is the value which depicted in the bottom right corner of the pink grid.

In the below figure, the Kernel moves to the right with a defined value for “Stride.” Along the way, it parses the image objects until it completes the breadth. Then it hops down to the second row on the left and moves just as in the top row till it covers the whole image. The process keeps repeating until every part of the image is parsed.

![kernel_parse](cv_algorithms/images/parse.png)

If there are multiple channels such as found in RGB images, then the kernel contains the same depth as found in the input image. The multiplication of the matrix is implemented based on the number of Ks. The procedure is followed as in *stack* format, for example, {K1, I1}, {K2, I2}, and so on. The results are generated based on the summation of bias. The result is in the form of a squeezed “1-depth channel” of convoluted feature output.

The goal of this convolution operation is to obtain all the high-level features of the image. The high-level features can include edges of the image too. This layer is not just limited to high-level features; it also performs an operation on low-level features, such as color and gradient orientation. This architecture evolves to a new level and thus includes two more types of layers. The two layers are known as Valid padding and the Same padding.

The objective of these layers is to reduce the dimensionality of the image that is found in the original input image and to increase dimensionality or, in some cases, to leave it unchanged, depending on the required output. The same padding is applied to convolute the image to different dimensions of the matrix, while valid padding is applied when there is no need to change the dimension of the matrix.

![kernel_channel](cv_algorithms/images/kernel-channel.png)

Pooling layer:

As identical to the recognised layer “convolutional,” the foremost aim of the Pooling layer is essential to decrease the spatial size of the Convolved Feature. So, in short words, it works for decreasing the required computational power for the processing of data by the method of dimensionality reduction. Moreover, it is also beneficial for the extraction of the dominant features, which are basically rotational as well as positional invariant, so the maintenance of the process effectively is needed.

Types of Pooling:

There are mainly two different types of Pooling which are as follows:

Max Pooling: ****The Max Pooling basically provides the maximum value within the covered image by the Kernel.

Average Pooling: The Average Pooling provides and returns the average value within the covered image by the Kernel.

![pooling](cv_algorithms/images/pooling.png)

The other functionality of Max Pooling is also noise-suppressing, as it works on discarding those activations which contain noisy activation. And on the other side, the Average Pooling simply works on the mechanism of noise-suppressing by dimensionality reduction. So, in short words, we can conclude that Max Pooling works more efficiently than Average Pooling.

The Convolutional Layer, altogether with the Pooling layer, makes the “i-th layer” of the Convolutional Neural Network. Entirely reliant on the image intricacies, the layer counts might be rise-up for the objective of capturing the details of the detailed level, but also needs to have more computational power. After analysing the above-described information about the process, we can easily execute the model for understanding the features. Moreover, here we are about to get the output and then provide it as an input for the regular Neural Network for further classification reasons.

Classification: Fully Connected Layer (FC Layer)

The addition of the FC layer is mostly the easiest way for the learning purpose of the non-linear combinations of the abstract level structures, as it is also revealed by the output of the convolutional layer. The FC layer provides the space for learning non-linear functions. As now we have achieved our task to convert our image output into a specific form of Multi-layer Perceptron, now we must flatten the output image into a form of a column vector. Over the different eras of epochs, the model is basically succeeded for the distinguishing function between the dominating and low-level features.

**CNNs Applications**

The best of the deep learning image classification models were created for the needs of the [ImageNet competition](https://www.image-net.org/index.php). ImageNet is a visual Dataset that contains more than 15 million of labeled high-resolution images covering almost 22,000 categories. ImageNet Dataset is of high quality and that’s one of the reasons it is highly popular among researchers to test their image classification model on this dataset.

Here are some impressive examples of CNN architectures:

- AlexNet
- GoogLeNet
- ZFNet
- LeNet
- ResNet

All of those models could be found under [Keras applications](https://keras.io/api/applications/), for both training and transfer learning (the model is pre-trained - most commonly in ImageNet data and you can download the trained parameters)

##### *1.2.1.1.2 Vision Transformers (ViT)*

In 2022, the Vision Transformer (ViT) emerged as a competitive alternative to convolutional neural networks (CNNs) that are currently state-of-the-art in computer vision and therefore widely used in different image recognition tasks. They were induced with this [paper](https://arxiv.org/abs/2010.11929). ViT models outperform the current state-of-the-art (CNN) by almost x4 in terms of computational efficiency and accuracy.

**Vision Transformer (ViT) in Image Recognition**

In computer vision, attention is either used in conjunction with convolutional networks (CNN) or used to substitute certain aspects of convolutional networks while keeping their entire composition intact. However, this dependency on CNN is not mandatory, and a pure transformer applied directly to sequences of image patches can work exceptionally well on image classification tasks.

Recently, Vision Transformers (ViT) have achieved highly competitive performance in benchmarks for several computer vision applications.

**Difference between CNN and ViT (ViT vs. CNN)**

Vision Transformer (ViT) achieves remarkable results compared to convolutional neural networks (CNN) while obtaining fewer resources for  pre-training. In comparison to convolutional neural networks (CNN), Vision Transformer (ViT) show a generally weaker inductive bias resulting in increased reliance on model regularisation or data augmentation when training on smaller datasets.

The ViT is a visual model based on the architecture of a transformer originally designed for text-based tasks. The ViT model represents an input image as a series of image patches, like the series of word embeddings used when using transformers to text, and directly predicts class labels for the image. ViT exhibits an extraordinary performance when trained on enough data, breaking the performance of a similar state-of-art CNN with 4x fewer computational resources.

These transformers have high success rates when it comes to NLP models and are now also applied to images for image recognition tasks. CNN uses pixel arrays, whereas ViT splits the images into visual tokens. The visual transformer divides an image into fixed-size patches, correctly embeds each of them, and includes positional embedding as an input to the transformer encoder. Moreover, [ViT models outperform CNNs](https://arxiv.org/abs/2105.07581) by almost four times when it comes to computational efficiency and accuracy.

The self-attention layer in ViT makes it possible to embed information globally across the overall image. The model also learns on training data to encode the relative location of the image patches to reconstruct the structure of the image.

The transformer encoder includes:

- Multi-Head Self Attention Layer (MSP): This layer concatenates all the attention outputs linearly to the right dimensions. The many attention heads help train local and global dependencies in an image.
- Multi-Layer Perceptrons (MLP) Layer: This layer contains a two-layer with Gaussian Error Linear Unit (GELU).
- Layer Norm (LN): This is added prior to each block as it does not include any new dependencies between the training images. This thereby helps improve the training time and overall performance.

Moreover, residual connections are included after each block as they allow the components to flow through the network directly without passing through non-linear activations.

In the case of image classification, the MLP layer implements the classification head. It does it with one hidden layer at pre-training time and a single linear layer for fine-tuning.

![ViT](cv_algorithms/images/vit.jpeg)

Raw images (left) with attention maps of the ViT-S/16 model (right). 

What are attention maps of ViT?

Attention, more specifically, self-attention is one of the essential blocks of machine learning transformers. It is a computational primitive used to quantify pairwise entity interactions that help a network to learn the hierarchies and alignments present inside input data. Attention has proven to be a key element for vision networks to achieve higher robustness.

![map](cv_algorithms/images/map.jpeg)

Vision Transformer ViT Architecture

The overall architecture of the vision transformer model is given as follows in a step-by-step manner:

1. Split an image into patches (fixed sizes)
2. Flatten the image patches
3. Create lower-dimensional linear embeddings from these flattened image patches
4. Include positional embeddings
5. Feed the sequence as an input to a state-of-the-art transformer encoder
6. Pre-train the ViT model with image labels, which is then fully supervised on a big dataset
7. Fine-tune the downstream dataset for image classification

![ViT_architecture](cv_algorithms/images/vit_arch.png)

While the ViT full-transformer architecture is a promising option for vision processing tasks, the performance of ViTs is still inferior to that of similar-sized CNN alternatives (such as ResNet) when trained from scratch on a mid-sized dataset such as ImageNet.

**How does a Vision Transformer (ViT) work?**

The performance of a vision transformer model depends on decisions such as that of the optimizer, network depth, and dataset-specific hyperparameters. Compared to ViT, CNNs are easier to optimize.

The disparity on a pure transformer is to marry a transformer to a CNN front end. The usual ViT stem leverages a 16*16 convolution with a 16 stride. In comparison, a 3*3 convolution with stride 2 increases the stability and elevates precision.

CNN turns basic pixels into a feature map. Later, the feature map is translated by a tokenizer into a sequence of tokens that are then inputted into the transformer. The transformer then applies the attention technique to create a sequence of output tokens. Eventually, a projector reconnects the output tokens to the feature map. The latter allows the examination to navigate potentially crucial pixel-level details. This thereby lowers the number of tokens that need to be studied, lowering costs significantly.

Particularly, if the ViT model is trained on huge datasets that are over 14M images, it can outperform the CNNs. If not, the best option is to stick to [ResNet](https://viso.ai/deep-learning/resnet-residual-neural-network/) or EfficientNet. The vision transformer model is trained on a huge dataset even before the process of fine-tuning. The only change is to disregard the MLP layer and add a new D times KD*K layer, where K is the number of classes of the small dataset.

To fine-tune in better resolutions, the 2D representation of the pre-trained position embeddings is done. This is because the trainable liner layers model the positional embeddings.

**Vision Transformer (ViT) Applications**

Vision transformers have extensive applications in popular [image recognition](https://viso.ai/computer-vision/image-recognition/) tasks such as [object detection](https://viso.ai/deep-learning/object-detection/), [segmentation](https://viso.ai/deep-learning/image-segmentation-using-deep-learning/), [image classification](https://viso.ai/computer-vision/image-classification/), and action recognition. Moreover, ViTs are applied in generative modeling and multi-model tasks, including visual grounding, visual-question answering, and visual reasoning.

Video forecasting and activity recognition are all parts of video processing that require ViT. Moreover, image enhancement, colorization, and image super-resolution also use ViT models. Last but not least, ViTs has numerous applications in 3D analysis, such as segmentation and point cloud classification. 

A very handy implementation of the ViT could be found [here](https://huggingface.co/docs/transformers/index). 

##### *1.2.1.1.3 ConvNext (CNNs are dead, long live the CNNs)*

A recent [research](https://arxiv.org/abs/2201.03545) claims that by borrowing ideas from the successes of the Vision transformer and CNNs, one can build a pure CNN whose performance match state-of-the-art-models like the Vision transformer. This was the idea behind ConvNext. For more details regarding the architecture of the model please have a look [here](https://medium.com/augmented-startups/convnext-the-return-of-convolution-networks-e70cbe8dabcc).

A vary straight-forward implementation of ConvNext could be found [here](https://huggingface.co/facebook/convnext-tiny-224).

#### **1.2.2 Image Segmentation**

Image segmentation is a method in which an image is broken down into various subgroups called Image segments which helps in reducing the complexity of the image to make further processing or analysis of the image simpler. Segmentation in easy words is assigning labels to pixels. All picture elements or pixels belonging to the same category have a common label assigned to them.

![segmentation](cv_algorithms/images/segmentation.png)

There are 4 major types of image segmentation (actually the types are 3 but Image Recognition which is more like a classification technique is included here in order to have an holistic overview):

1. **Image Recognition**

Image recognition refers to technologies that identify places, logos, people, objects, buildings, and several other variables in digital images. The computer sees an image as numerical values of pixels and in order to recognise a certain image, it has to recognise the patterns and regularities in them. In the image recognition we try to classify the image into various categories.

![recognition](cv_algorithms/images/recognition.png)

2. **Semantic Segmentation**

Semantic segmentation is the task of clustering parts of an image together which belong to the same object class. It is a form of pixel-level prediction because each pixel in an image is classified according to a category.

![semantic](cv_algorithms/images/semantic.png)

3. **Object Detection**

Object detection is a computer vision technique that works to identify and locate objects within an image or video. Specifically, object detection draws bounding boxes around these detected objects, which allow us to locate where said objects are in a given image.

Object detection is commonly confused with image recognition, so before we proceed, it’s important that we clarify the distinctions between them. Image recognition assigns a label to an image. A picture of a dog receives the label “dog”. A picture of two dogs, still receives the label “dog”. Object detection, on the other hand, draws a box around each dog and labels the box “dog”. The model predicts where each object is and what label should be applied. In that way, object detection provides more information about an image than recognition.

![object](cv_algorithms/images/object.png)

4. **Instance Segmentation**

Instance Segmentation is the technique of detecting, segmenting, and classifying every individual object in an image. We can refer to Instance Segmentation as a combination of semantic segmentation and object detection (detecting all instances of a category in an image) with the additional feature of demarcating separate instances of any particular segment class added to the vanilla segmentation task. Instance Segmentation produces a richer output format as compared to both object detection and semantic segmentation networks.

![instance](cv_algorithms/images/instance.png)

##### **1.2.2.1 Object Detection / Image Segmentation algorithms**

##### *1.2.2.1.1 R-CNN*

[Ross Girshick et al](https://arxiv.org/pdf/1311.2524.pdf) proposed a method where we use selective search to extract just the regions from the image and he called them region proposals. Therefore, now, instead of trying to classify the whole image, you can just work with selected regions. These region proposals are generated using the selective search algorithm.

![RCNN](cv_algorithms/images/rcnn.png)

An implementation in python for R-CNN could be found [here](https://towardsdatascience.com/step-by-step-r-cnn-implementation-from-scratch-in-python-e97101ccde55).

Even though the R-CNN improved the object detection procedures they had significant drawbacks like:

- It still takes a huge amount of time to train the network as you would have to classify a lot of region proposals per image.
- It cannot be implemented real time as it takes around 47 seconds for each test image.
- The selective search algorithm is a fixed algorithm. Therefore, no learning is happening at that stage. This could lead to the generation of bad candidate region proposals.

##### *1.2.2.1.2 Fast R-CNN*

The same author of the previous paper(R-CNN) solved some of the drawbacks of R-CNN (with [this paper](https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Girshick_Fast_R-CNN_ICCV_2015_paper.pdf)) to build a faster object detection algorithm and it was called Fast R-CNN. The approach is similar to the R-CNN algorithm but instead of feeding the region proposals to the CNN, we feed the input image to the CNN to generate a convolutional feature map. From the convolutional feature map, we identify the region of proposals and warp them into squares and by using a RoI pooling layer we reshape them into a fixed size so that it can be fed into a fully connected layer. From the RoI feature vector, we use a softmax layer to predict the class of the proposed region and also the offset values for the bounding box.

The reason “Fast R-CNN” is faster than R-CNN is because you don’t have to feed all region proposals to the convolutional neural network every time. Instead, the convolution operation is done only once per image and a feature map is generated from it.

![Fast](cv_algorithms/images/fast.png)

[Here](https://github.com/rbgirshick/fast-rcnn) could be found a handy implementation of Fast R-CNN.

Even though the Fast R-CNN showed a significant improvement time wised, still the time needed was not appropriate for making this technique a real time. Also the time showed a significant increase when the region proposal was included.

##### *1.2.2.1.3 Faster R-CNN*

Both of the above algorithms(R-CNN & Fast R-CNN) uses selective search to find out the region proposals. Selective search is a slow and time-consuming process affecting the performance of the network. Therefore, [Shaoqing Ren et al](https://arxiv.org/pdf/1506.01497.pdf). came up with an object detection algorithm that eliminates the selective search algorithm and lets the network learn the region proposals.

Similar to Fast R-CNN, the image is provided as an input to a convolutional network which provides a convolutional feature map. Instead of using selective search algorithm on the feature map to identify the region proposals, a separate network is used to predict the region proposals. The predicted region proposals are then reshaped using a RoI pooling layer which is then used to classify the image within the proposed region and predict the offset values for the bounding boxes.

![Faster](cv_algorithms/images/faster.png)

A great Keras implementation for Faster R-CNN could be found [here](https://towardsdatascience.com/faster-r-cnn-object-detection-implemented-by-keras-for-custom-data-from-googles-open-images-125f62b9141a).

##### *1.2.2.1.4 Mask R-CNN*

[Mask R-CNN](https://arxiv.org/abs/1703.06870) is a Convolutional Neural Network (CNN) and state-of-the-art in terms of image segmentation and instance segmentation. Mask R-CNN was developed on top of Faster R-CNN. While Faster R-CNN has 2 outputs for each candidate object, a class label and a bounding-box offset, Mask R-CNN is the addition of a third branch that outputs the object mask. The additional mask output is distinct from the class and box outputs, requiring the extraction of a much finer spatial layout of an object.

Mask R-CNN, as an extension of Faster R-CNN, works by adding a branch for predicting an object mask (Region of Interest) in parallel with the existing branch for bounding box recognition.

Advantages of Mask R-CNN

- Simplicity: Mask R-CNN is simple to train.
- Performance: Mask R-CNN outperforms all existing, single-model entries on every task.
- Efficiency: The method is very efficient and adds only a small overhead to Faster R-CNN.
- Flexibility: Mask R-CNN is easy to generalize to other tasks.

The key element of Mask R-CNN is the pixel-to-pixel alignment, which is the main missing piece of Fast/Faster R-CNN. Mask R-CNN adopts the same two-stage procedure with an identical first stage (which is RPN). In the second stage, in parallel to predicting the class and box offset, Mask R-CNN also outputs a binary mask for each RoI. This is in contrast to most recent systems, where classification depends on mask predictions.

Furthermore, Mask R-CNN is simple to implement and train given the Faster R-CNN framework, which facilitates a wide range of flexible architecture designs. Additionally, the mask branch only adds a small computational overhead, enabling a fast system and rapid experimentation.

![Mask](cv_algorithms/images/mask.png)

##### *1.2.2.1.5 YOLO (You Only Look Once)*

All of the previous object detection algorithms use regions to localize the object within the image. The network does not look at the complete image. Instead, parts of the image which have high probabilities of containing the object. [YOLO or You Only Look Once](https://arxiv.org/abs/1506.02640) is an object detection algorithm much different from the region based algorithms seen above. In YOLO a single convolutional network predicts the bounding boxes and the class probabilities for these boxes.

![YOLO](cv_algorithms/images/yolo.png)

How YOLO works is that we take an image and split it into an SxS grid, within each of the grid we take m bounding boxes. For each of the bounding box, the network outputs a class probability and offset values for the bounding box. The bounding boxes having the class probability above a threshold value is selected and used to locate the object within the image.

YOLO is orders of magnitude faster(45 frames per second) than other object detection algorithms. The limitation of YOLO algorithm is that it struggles with small objects within the image, for example it might have difficulties in detecting a flock of birds. This is due to the spatial constraints of the algorithm.