---
layout: post
title: Better attention with Performers?
# subtitle: A post on fariness issues in credit card fraud detection algorithms
# cover-img: /assets/img/gengis/back.jpg
thumbnail-img: /assets/img/performer/img1.png
share-img: /assets/img/performer/img1.png
tags: [ NLP, Machine Learning, tech]
---

Just last month Google released a paper introducing Performers called [Rethinking Attention With Performers](https://arxiv.org/pdf/2009.14794.pdf). It inroduces a new Transformer architecture which estimates the full-rank attention mehcanism to approximate softmax kernel with linear space & time complexity.

Soon after it's release I came across this amazing blog post by Teddy Kokker, [Performers: The Kernel Trick, Random Fourier Features, and Attention](https://teddykoker.com/2020/11/performers/). This very informative article inspired me to gather some of my own thoughts on this in a space. I would highly recommend reading Teddy's blog post.

### Transformer & Sequence Lengths ###

Self attention mechanism, a defining feature of the Transformer architecture was introduced in the *Transformer* paper[(Vaswani et al.,2017)](https://teddykoker.com/2020/11/performers/#vaswani2017attention).
We can define attetion mathematically as :
<!-- ![](/assets/img/performer/img2.png =50) -->
<img src='/assets/img/performer/img2.png' height=50>

Where the Q,K,V variables are the Query,Key and Value vectors. They are projections of each word in the input sequence. They are abstractions that are useful for calculating and thinking about attention. For a more detailed explanation of these vectors I'd recommend going through this [amazing post](http://jalammar.github.io/illustrated-transformer/).
We know the vectors:

 <img src="https://latex.codecogs.com/svg.latex?\Large&space;Q,K,V\in\mathbb{R}^{L\times d}" title="\Large Q,K,V\in\mathbb{R}^{L\times d}" height=20/>

 Where *L* is the sequence lenght and *d* is a hidden dimension. If we expand the softmax we will notice our attention matrix, *A* will have a shape of *LxL*, this means any operation performed with A will have a time and space complexity that grows quadratically w.r.t sequence length *L*. This means Transformer architecture have a limitation on sequence lengths which can be used and are thus not suitbale for tasks requiring larger sequence lengths.

There have already been many efforts to solve this issue with Transformers like [memory cachine techniques](https://ai.googleblog.com/2019/01/transformer-xl-unleashing-potential-of.html) and [sparse attention](https://openai.com/blog/sparse-transformer/). Sparse attention reduce the time and memory complexity by only computing a limited selection of similarity scores from a sequence than all pairs, thus resulting in a sparse attention matrix. These sparse matrices can be manually made,learned or found by other methods.Such sparsity-based architectures usually require additional layers to implicitly produce a full attention mechanism. 

According to this Google AI [blog post](https://ai.googleblog.com/2020/10/rethinking-attention-with-performers.html), sparse matrices have a lot of issues still. Like, reuqiring efficient sparse-matrix multiplications,optimized primarily for Transformers and generative pre-training, they stack more attention layers to compensate for sparsity making them difficult to be used in pre-trained models. In addition there are still many opeations like softmax which can't be sparsified.


### Enter Performer ###

So how does Performer solve this problem of large attention matrix limiting the sequence lengths?
Performer introduces Generalized Attention in a Transformer architecture ,where attention scales linearly, thus enabling faster training while allowing model to process longer lengths. 

The performer does this by using a **softmax kernel** and [random Fourier features](https://people.eecs.berkeley.edu/~brecht/papers/07.rah.rec.nips.pdf) introduced in a very popular paper at NIPS. The softmax kernel is based on the familiar kernel trick we see when we use SVMs. 
There is a lot of complex math here,so I'll source some of it from [Teddy's blog](https://teddykoker.com/2020/11/performers/#rahimi2007random) and would advise you too look up some formulae there.

But in short, using the paper on Random Fourier Feature and kernel trick, we can rewrite the attentio matrix equation using a softmax kernel as:

<img src="https://latex.codecogs.com/svg.latex?\Large&space;A=exp(QK^T)....(i)" title="\Large eq2" height=20/>

<img src="https://latex.codecogs.com/svg.latex?\Large&space;K_{softmax}(x_i,x_j)=exp(x_i^Tx_j^T)....(ii)" title="\Large eq4" height=25/>

<img src="https://latex.codecogs.com/svg.latex?\Large&space;A(i,j)=K_{softmax}(q_i^T,k_j^T)....(iii)" title="\Large eq3" height=25/>

Where q_i and k_j are the ith and jth row vectors in Q and K. Following the paper we can approximate the softmax kernel function to a lower dimensionality as they do for Gaussian kernel.

This approximation allows the Attention matrix to go smaller in size, from LxL dimensions to RxL dimension.As the sum of each row approximates it's full-rank counterpart.

#### Other Improvements ####
There are also some other improvements in the Performer paper like using orthogonal random features instead of IID features and also positive random features. These are much better summarized in this [blog](https://ai.googleblog.com/2020/10/rethinking-attention-with-performers.html)


### Conclusion ###

The Performer paper has posted some results comparing the results vs the transformer.

|![](/assets/img/performer/img4.png)|
| :--: |
|*Performance modelling protein sequences. Source:[Link](https://ai.googleblog.com/2020/10/rethinking-attention-with-performers.html)*|

There is definitely a clear improvement in the accuracy not to mention even in the train times given the smaller A matrix. Going through this paper and some of the great blogs about it has really improved my understanding of Attention and the Transfomer architrecutre a lot. Not to say also of kernel functions , which I had kinf of forgotten about since my first ML course and SVMs. 

It was agreat learning experience diving into this very exciting improvement over transformers and recent efforts in sparse attention models. It would be really interesting to see how many Performer models we might see in the big conference.