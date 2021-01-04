---
layout: post
title: Content Aware Image Resizing with Deep Learning
# subtitle: A post on fariness issues in credit card fraud detection algorithms
# cover-img: /assets/img/gengis/back.jpg
thumbnail-img: /assets/img/performer/img1.png
share-img: /assets/img/performer/img1.png
tags: [ Machine Learning, Computer Vision, tech]
---

This post is about how we can combine classical and deep learning approaches to content aware image resizing.

Images must often be resized to fit different display sizes. So what's the challenge with this ? Normally whenever we resize image, it can lead to chagning of aspect ratios and when we freeze aspect ratio, it can lead to content of the image being too small.

![](/assets/img/carving/img1.png)

We can see those issues in the above image. So the right way to solve this would be to resize and crop the image in such a way that only the right information stays. This way we do not have to deal with aspect ratios. And as it happens a classical computer vision approach exists to help us with this issue.

### Seam Carving ###

Seam carving is a simple image operator that supports content aware image resizing for both reduction and expansion. It was introduced in this [paper](http://graphics.cs.cmu.edu/courses/15-463/2012_fall/hw/proj3-seamcarving/imret.pdf)

### What Is a Seam ###

A seam is an optimal 8-connected path of pixels on a single image from top to bottom, or left to right, where optimality is defined by an image energy function. By repeatedly carving out or inserting seams in one direction we cna change the aspect ratio of the image. By applying these operators in both directions we can retarget the image to a new size.The selection and order of seams protect the content of the image,as defined by the energy function. Seam carving can also be usedfor image content enhancement and object removal. By storingthe order of seams in an image we create multi-size images, that areable to continuously change in real time to fit a given size. 

#### Energy Functions ####

Seam carving works based on energy functions and operators to help us decice the criteria for remving pixels. The paper talks about a few energy functions , but singles out two of the to perform better.

First : <img src='/assets/img/carving/img3.png' width = 150>
Second: <img src='/assets/img/carving/img4.png' width = 180>

Where I stands for pixel intensity and we are looking at their gradients.For the second function HoG(I(x,x)) is taken to be a histogram of gradietns of every pixel around a 11x11 window.

Now given an energy function our task is to form strategies to remove pixels based on this. For example removing pixels with lowest energy in ascending order. If we want to prevent the image frombreaking we can remove an equal number of low energy pixels from every row. This preserves the rectangular shape of the image but de-stroys the image content by creating a zigzag effect

Below is an example of seam carving in action, removing seams vertically. Seams can also be removed horizontally depending on how we want to resize the image.
![](/assets/img/carving/img2.gif)
*Source: Andrew Campbell, [GitHub](https://github.com/andrewdcampbell)*

These are basic principles of seam carving while there are multiple strategies which is covered in the paper. 

Thankfully for us a great and very easy to use implementation of seam carving was created by Andrew Campbell located on his GitHub [here](https://github.com/andrewdcampbell). I used this for all my experiments.

#### Results From Seam Carving ####

Seam carving performs really well for a lot of scenarios. Whether it be increasing or decreasing image size.
But it sometimes fails , and distorts the contents of the image. You can see some of the results below:

![](/assets/img/carving/img5.png)

Since we see that there has to be some sort of content awareness to make seam carving perform best, we look at another approach.


### Resizing Using Saliency Maps ###

Another way to approach resizing using deep learning is to find important regions in the image. Using [saliency maps](https://en.wikipedia.org/wiki/Saliency_map) is a common way to achieve that. Saliency estimation is a very common methid in image segmentation and there are a lot of existing methods to do that.

However in an image not all features are important for saliency , hence we can resort to attention mechanisms.There a lot of existing methods for this task like SalNet,progressive attention networks ,stacked attention networks etc.We implement a feature attention network to capture the important features.We implement the approach on the following paper which was accepted in CVPR last year : [Pyramid Feature Attention Networks for Saliency Maps](https://arxiv.org/pdf/1903.00179v2.pdf) and the code is open sourced [here](https://github.com/CaitinZhao/cvpr2019_Pyramid-Feature-Attention-Network-for-Saliency-detection)

#### Network Architecture ####

This is a unique architecture to get saliency maps proposed in the [paper](https://arxiv.org/pdf/1903.00179v2.pdf) above. 
This is what the architecture looks like :

![](/assets/img/carving/img6.png)
*[Source](https://arxiv.org/pdf/1903.00179v2.pdf)*

The CNN network is VGG-16 pre trained on ImageNet.The paper also proposes a uniques pyramid feature attentio network, consisting of channel-wise and spatial attention. In short the channel wise attention for high level features which help us get rich context information. Spatial attention for low level features, as they help us differentiate background and foreground.

![](/assets/img/carving/img7.png)
*[Source](https://arxiv.org/pdf/1903.00179v2.pdf)*


#### Training the Saliency Network ####

I trained the above arhcitecture on the [MSRA10k dataset](https://mmcheng.net/msra10k/). This dataset is a more fine grained version of the MSRA dataset with more accurate pixel level saliency labeling.
I used a weighted cross entropy as our loss function.I’ve used the following data augmentations for robustness : Random rotation, Random crop, Random brightness change, Random horizontal flip
With a batch size 22 , image size of 256x256 running about 40 epochs we get our F1 score to be : 0.8528

### Completing the Pipeline ###

Once we get the saliency maps, we move on to perform a few more steps to get the final results. This how our complete pipeline looks like :

![](/assets/img/carving/img8.png)

We take our generated saliency maps, and use binary thresholding to smoothen our maps followed by dilation. Once we have the dilated maps we then find the contours of the generated regions which we then use to crop out the important content.

#### Results with Seam Carving ####

I got pretty great results from saliency maps. Below are the results for the same images we use for our seam carving example.

![](/assets/img/carving/img9.png)

### Conclusion ###

I observed that seam carving can distort objects but is a more efficient and better approach
where there are lesser objects. Saliency maps are better with lot of actors being present in the scene.But, saliency maps won’t work if we have to increase the size of the image. Hence what we suggest is a hybrid approach :
- If we have to increase size of the image , we can go for seam based approach.
- Otherwise :
    - Get the saliency map of the image.
    - Identify the seams which may pass through our maps and avoid them.

All the code used for this article can be found [here](https://github.com/rajat08/Image-Resizing-with-Saliency-Maps)