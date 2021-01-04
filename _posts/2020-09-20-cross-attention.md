---
layout: post
title: Getting Cross Attention Weights for Hugging Face Transformers
subtitle: A little-hacky method to get encoder-decoder attention from Transformers
# cover-img: /assets/img/path.jpg
thumbnail-img: /assets/img/cross-attention/arch.jpg
share-img: /assets/img/cross-attention/arch.jpg
tags: [NLP, Machine Learning, tech]
---

I was recently involved in a research project where we were trying a model-based active learning method in Neural Machine Translation, which utilizes the multi-headed multi-layered encoder-decoder attention weights of the Transformer architecture. 

It was a little troublesome for us to extract the encoder-decoder attention weights for the [HuggingFace](https://huggingface.co/) models that we were using. Hence I’ve decided to just write it down how I got it done, since I found no resources online for it as of the time of writing. Hopefully it will save someone some effort and frustration in the future.

To give you a context of where this component lies , check the image below from the very popular and amazing [article](http://jalammar.github.io/illustrated-transformer/) on Transformers by Jay Alammar on his blog.

|![](/assets/img/cross-attention/arch.jpg)|
| :--: |
|*Source: http://jalammar.github.io/illustrated-transformer/*|

We decided on using [Marian NMT](https://marian-nmt.github.io/) for our task. And like most NLP projects these days ,we decided to use HugginFace’s implementation for the model. The models were originally trained by [Jörg Tiedemann of Helsinki University](https://researchportal.helsinki.fi/en/persons/j%C3%B6rg-tiedemann).

## The Problem

The Marian NMT implementation in question , reuses most of the it’s [code](https://huggingface.co/transformers/_modules/transformers/models/marian/modeling_marian.html#MarianMTModel) from the [BartForConditionalGeneration implementation](https://huggingface.co/transformers/_modules/transformers/models/bart/modeling_bart.html#BartForConditionalGeneration) . While the transformers library does allow us to view attention weights by setting the flag : output_attentions = True

There is one issue , this only returns the self attentions weights for the encoder and decoder. If you look closely at the attention class, you would see that while encoder-decoder attentions are calculated, they are not returned during the forward and hence added to the vector containing rest of the information for the output_attentions flag.

## The Solution

The solution to this problem is pretty straightforward. All the code changes we make will be in just one file. You can find it [here](https://drive.google.com/file/d/1w67zs3NFma1Uj5meAMuVx1JBCskudVCe/view?usp=sharing).

Since we now want the enocoder-decoder attention to be returned, we will make changes in the BartDecoder file allowing it to return this extra parameter from decoder_layer object :

```
x, layer_self_attn, layer_past, enc_dec_attn = decoder_layer(x,
                                                            encoder_hidden_states,
                                                            encoder_attn_mask=encoder_padding_mask,decoder_padding_mask=decoder_padding_mask,
                                                            layer_state=layer_state,
                                                            causal_mask=decoder_causal_mask,output_attentions=output_attentions,)
if use_cache:    
    next_decoder_cache.append(layer_past.copy())
if output_attentions:
    all_self_attns += (layer_self_attn,)
    enc_dec_all_attn += (enc_dec_attn,)#return the enc_dec_all_attn

return enc_dec_all_attn
```

Now all you need to do is to import this modified file to your notebook and when you now call the forward wiht output_attentions=True you will get the attention weights back.

In case you have any confusion regarding this please refer to this [file](https://drive.google.com/file/d/1w67zs3NFma1Uj5meAMuVx1JBCskudVCe/view?usp=sharing). Also you can take a look at my [notebook here](https://github.com/rajat08/CS591W1_project/blob/master/Experiments/Marian_MT/scoring-heads.ipynb) where we use this method on the file marian_mt.py and apply our scoring criteria.