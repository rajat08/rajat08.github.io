---
layout: post
title: Sentiment Analysis of Airline Tweets Using BERT and InterpretML
subtitle: A little post on some of the work I did during my Red Hat internship
# cover-img: /assets/img/path.jpg
thumbnail-img: /assets/img/sentiment-analysis/img_3.png
share-img: /assets/img/sentiment-analysis/img_3.png
tags: [NLP, Machine Learning, tech]
---

Sentiment analysis is the automated process to analyze a text and interpret the sentiments behind it. Through machine learning and text analytics, algorithms can classify statements as positive, negative, and neutral.

Sentiment analysis is a very popular NLP task used to analyze social media posts, tweets, and online product reviews, as a way to track opinions, reactions, and ultimately improve customer service and experience. It’s great for market research, brand and product reputation monitoring, and customer experience analysis.

In this blog I’ll run through a simple workflow of using BERT to perform sentiment analysis on text data. Then we will see how can we use interpretability techniques to understand how BERT makes decisions.

## Problem Statement  ##

The goal is to make a deep learning model which can classify emotion in a given sentence.

We do this by making use of transfer learning on the BERT model architecture. So that this can be used as a sample workflow, we take publicly available data as an example, as the original workflow consists of sensitive data. We also discuss interpreting BERT using the Unified Information Explainer algorithm.

## Methodology ##

Given a text our goal is to predict whether it conveys a positive, negative or neutral emotion. Hence we want to build a text classifier for our data. There are various approaches to perform this task but for our project we pick the approach used in most state-of-the-art textual analysis systems i.e. deep learning.

To construct a deep learning model which is very accurate we require huge amounts of data and compute resources. But luckily for us models like BERT are pre trained on large amounts of data and made publicly available. Therefore we can fine tune an already pre trained model like [BERT](https://ai.googleblog.com/2018/11/open-sourcing-bert-state-of-art-pre.html) on our own data to leverage what the model has already learnt .This process is called transfer learning.

## BERT implementation ##

We are going to use Google’s pre trained BERT for our classification tasks. Apart from the model itself we also directly use Google’s scripts to run our classifier which enables us to use the model for our data specifically.

### Exploring the Dataset ###

To work our way through the pipeline we make use of the [Twitter US Airline Sentiment dataset](https://www.kaggle.com/crowdflower/twitter-airline-sentiment) . This dataset consists of tweets directed at six US airlines with each of them classified into neutral, positive or negative.

### Loading and Cleaning data ###

First we load up our data in the csv format into a pandas dataframe. We have the following columns in our dataset :
![](/assets/img/sentiment-analysis/img_1.png)
Features in the airline tweets data set

Since we are concerned with just the text and classifying based on that , we our only interested in text and airline_sentiment columns. Of course for a lot of datasets it’s possible and advised to look at multiple features to augment the decision making process but for this blog we are sticking to basic sentiment analysis of text.

If we look through the text columns , this is what the data looks like :
```
0    @VirginAmerica What @dhepburn said.                                                                                           
1    @VirginAmerica plus you've added commercials to the experience... tacky.                                                      
2    @VirginAmerica I didn't today... Must mean I need to take another trip!                                                       
3    @VirginAmerica it's really aggressive to blast obnoxious "entertainment" in your guests' faces &amp; they have little recourse
4    @VirginAmerica and it's a really big bad thing about it                                                                       
Name: text, dtype: object
```

And if we plot it out , this is how the data is divided :
![](/assets/img/sentiment-analysis/img_2.png)
Although the data contains a much larger percentage of negative tweets, the other categories still have enough data in them. Hence we don’t have to perform any undersampling/oversampling operations.

We also perform some pre-processing to clean our data like getting rid of special characters, removing single characters which provide no value to us, eliminating extra spaces.

### Preprocessing tweets ###

We perfrom some basic cleaning on our text data using regular expressions. We then split our data into test and training sets.

```
from sklearn.model_selection import train_test_split
features = tweets.iloc[:, 10].values
labels = tweets.iloc[:, 1].values
#preprocessing 
processed_features = []

for sentence in range(0, len(features)):
    #Getting rid of special characters
    processed_feature = re.sub(r'\W', ' ', str(features[sentence]))
    # remove all single characters
    processed_feature= re.sub(r'\s+[a-zA-Z]\s+', ' ', processed_feature)
    # Remove single characters from the start
    processed_feature = re.sub(r'\^[a-zA-Z]\s+', ' ', processed_feature) 
    # Substituting multiple spaces with single space
    processed_feature = re.sub(r'\s+', ' ', processed_feature, flags=re.I)
    # Removing prefixed 'b'
    processed_feature = re.sub(r'^b\s+', '', processed_feature)
    # Converting to Lowercase
    processed_feature = processed_feature.lower()
    processed_features.append(processed_feature)

#Splitting the data 
X_train, X_test, y_train, y_test = train_test_split(processed_features, labels, test_size=0.2, random_state=0)

Mapping the emotions to numbers for the training and inference step

d = {"positive":2,"negative":0,"neutral":1}
y_train = [d[x] for x in y_train]
y_test = [d[x] for x in y_test]
```

### Applying Deep Learning using BERT ###

As mentioned before we would be using BERT and fine tune it to make predictions on our data.

The diagram below shows how BERT fits into our workflow.
![](/assets/img/sentiment-analysis/img_3.png)

Once we have cleaned our text data, all we have to do is to prepare it for consumption by the model. Depending on which implementation of BERT you want to use this step may differ. But all the approaches require us to encode our labels and tokenize the text. Both these functionalities are generally provided by the libraries offering the BERT implementation.

Since we just want to fine tune the model, we don’t have to put in a lot of resources in training. A couple of epochs are good enough to give us good results.

Loading the model:
```
folder = 'models/bert'
BERT_MODEL = 'uncased_L-12_H-768_A-12'
BERT_PRETRAINED_DIR = f'{folder}/uncased_L-12_H-768_A-12'
OUTPUT_DIR = f'{folder}/outputs'
print(f'>> Model output directory: {OUTPUT_DIR}')
print(f'>>  BERT pretrained directory: {BERT_PRETRAINED_DIR}')
```
### Training the model ###

Now that we have our data ready for use we move on the next step i.e training the model on our data.Since we already have the pre-learned weights on the model we can get good results by training the model on our data for just a few epochs.

We use the following hyper parameters for the training process :
```
TRAIN_BATCH_SIZE = 32
EVAL_BATCH_SIZE = 8
LEARNING_RATE = 1e-5
NUM_TRAIN_EPOCHS = 3.0
WARMUP_PROPORTION = 0.1
#We need this to be a little lower thant the max length of tweets we have 
MAX_SEQ_LENGTH = 50
# Model configs
SAVE_CHECKPOINTS_STEPS = 100000 #if you wish to finetune a model on a larger dataset, use larger interval
# each checpoint weights about 1,5gb
ITERATIONS_PER_LOOP = 100000
NUM_TPU_CORES = 8
```
Using the downloaded model :
```
train_features = run_classifier.convert_examples_to_features(
    train_examples, label_list, MAX_SEQ_LENGTH, tokenizer)
print('>> Started training at {} '.format(datetime.datetime.now()))
print('  Num examples = {}'.format(len(train_examples)))
print('  Batch size = {}'.format(TRAIN_BATCH_SIZE))
tf.logging.info("  Num steps = %d", num_train_steps)
train_input_fn = run_classifier.input_fn_builder(
    features=train_features,
    seq_length=MAX_SEQ_LENGTH,
    is_training=True,
    drop_remainder=True)
estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)
print('>> Finished training at {}'.format(datetime.datetime.now()))
```

### Predicting and Evaluating ###

Now that our training step is complete in the next steps we will use what our model learned to make predictions on the dataset. We will now evaluate our results.
```
def input_fn_builder(features, seq_length, is_training, drop_remainder):
  """Creates an `input_fn` closure to be passed to TPUEstimator."""

  all_input_ids = []
  all_input_mask = []
  all_segment_ids = []
  all_label_ids = []

  for feature in features:
    all_input_ids.append(feature.input_ids)
    all_input_mask.append(feature.input_mask)
    all_segment_ids.append(feature.segment_ids)
    all_label_ids.append(feature.label_id)

  def input_fn(params):
    """The actual input function."""
    print(params)
    batch_size = 500

    num_examples = len(features)

    d = tf.data.Dataset.from_tensor_slices({
        "input_ids":
            tf.constant(
                all_input_ids, shape=[num_examples, seq_length],
                dtype=tf.int32),
        "input_mask":
            tf.constant(
                all_input_mask,
                shape=[num_examples, seq_length],
                dtype=tf.int32),
        "segment_ids":
            tf.constant(
                all_segment_ids,
                shape=[num_examples, seq_length],
                dtype=tf.int32),
        "label_ids":
            tf.constant(all_label_ids, shape=[num_examples], dtype=tf.int32),
    })

    if is_training:
      d = d.repeat()
      d = d.shuffle(buffer_size=100)

    d = d.batch(batch_size=batch_size, drop_remainder=drop_remainder)
    return d

  return input_fnpredict_examples = create_examples(X_test, 'test')

predict_features = run_classifier.convert_examples_to_features(
    predict_examples, label_list, MAX_SEQ_LENGTH, tokenizer)

predict_input_fn = input_fn_builder(
    features=predict_features,
    seq_length=MAX_SEQ_LENGTH,
    is_training=False,
    drop_remainder=False)

result = estimator.predict(input_fn=predict_input_fn)
```
### Results ###

We get the following results for our model post training :
![](/assets/img/sentiment-analysis/img_4.png)

Our model doesn’t perform well when it comes to neutral sentiment. Possible reason for this could be the general ambiguity which comes in classifying a neutral emotion. Not to say the performance can’t be improved with some tweaking!
### Interpreting The Model ###

To better understand and improve our model we need some insights about how decisions are being made. One approach to do this is to use interpretability techniques.

For our case, we make use of the library [interpret-text](https://github.com/interpretml/interpret-text). As this library supports only PyTorch we will retrain our model using pytorch. We then use this trained BERT model to run our interpretability algorithm.

We use the [Unified Information Explainer](https://www.microsoft.com/en-us/research/publication/towards-a-deep-and-unified-understanding-of-deep-neural-models-in-nlp/) for the task.

With intepret-text it is fairly straightforward to train the explainer model. The only problem being that it accepts PyTorch models. Once you have a BERT model fine tuned on PyTorch , we can just use that as input to the explainer.

This is how we accomplish that :
```
interpreter_unified = UnifiedInformationExplainer(model=classifier.model,                                                         train_dataset=list(X_train),                                                         device=device,                                                         target_layer=14,                                                         classes=label_encoder.classes_)                                                                               idx = 1010                       
text = X_test[idx]                       
true_label = y_test[idx]                       
predicted_label = label_encoder.inverse_transform([preds[idx]])                       print(text, true_label, predicted_label)                                               explanation_unified = interpreter_unified.explain_local(text, true_label)                       
from interpret_text.experimental.widget import ExplanationDashboard                                               ExplanationDashboard(explanation_unified)
```
The way the dashboard works is that we can move our slider to pick the ’n’ most important features according to the model for making a certain prediction. It considers not just the word but it’s surrounding words as well.

For the example we are focusing on what the model sees at the final classification layer.

Sentence: @united yup it just happens way too often 5 times in the last 12 months

True Label: negative

Prediction: negative
![](/assets/img/sentiment-analysis/img_5.png)
The explainability dashboard

We see that our model focuses the most around the part ‘way too often’ and correctly predicts that it conveys a negative emotion.

We can effectively use this tool to look at a subset of sentences and tweak our model by looking at how it processes the sentences.
References

The source code related to the blog is located [here](https://github.com/aicoe-aiops/sentiment-analysis-public)

You can find the explainability notebook in the experimental folder.

