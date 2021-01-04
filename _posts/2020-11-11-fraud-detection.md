---
layout: post
title: Fairness and Biases in Credit Card Fraud Detection algorithms
subtitle: A post on fariness issues in credit card fraud detection algorithms
# cover-img: /assets/img/path.jpg
thumbnail-img: /assets/img/credit-card/img_2.png
share-img: /assets/img/credit-card/img_2.png
tags: [AI Fairness, MLOps, tech]
---

As a part of my fairness in AI class at Boston University, this fall 2020, I picked up the problem of where fairness issues creep in or can creep in when we design a Credit Card Fraud Detection Algorithm. 

I decided why not make a small blog post about it :)

## Problem Statement ##

Let's put ourselves in a more realistic situation if we were to assess this problem. Let us assume we are a data scientist working for a financial services company has to build and deploy a machine learning model. The model is trained on historical transactional data. And for each incoming transaction, the model is tasked to predict if it's fraudulent or not. Flagged transactions are then sent for further investigation by human analysts. 
Our objective is to critically assess any potential fairness or transparency issues that can come up if this model is used in the real-world.

## Stakeholders ##

From a standpoint of AI fairness being aware of the stakeholders in a project is a very important aspect. And as for this problem we can divide them into two categories :

**Business**

- The user making the payment
- Business(or other person) to whom the payment is made
- Banks, i.e. the 3rd party which authorizes and facilitates the transaction

**Development**

- Software Developers
-  QA Engineers
- Product Managers


## Potential Biases ##


#### Algorithmic Issues ####
Anomaly (or fraud) detection is different from a regular classification problem as it’s not possible to categorize anomalies in a separate class of its own. This is due to the fact that each anomaly can potentially be unique in its characteristics.
Due to this data imbalance, a separate family of anomaly detection algorithms are used in cases like these.
Also In such problems, there is a strong emphasis on not missing any fraudulent transactions which inadvertently leads to a higher rate of false positives (positive implies fraud). This phenomenon disproportionally affects the end-users more than the banks.

#### Data & Related Issues ####
When we start looking at the data related to the task of debit and credit fraud detection, we notice that there are a lot of inherent imbalance features due to the nature of the data.

First ,fraudulent transactions form a very small subset of all the data we receive , generally about 0.2% as observed in some studies ,which gives us a very small amount of data to base our models off .And if we look at some of the data that we typically get from a transaction , we can basically divide them into three broad categories transactional : relating to the info like amount , authentication and type Card holder : Country ,Age , Gender ,Name etc , gender may or may not be present depending on some situations and part from the holder’s profile we have info about their accounts like limit ,card type etc. There’s also merchant data present which along with various other data contains info about location and other profile related details.

![Data a credit card detection algorithm sees](/assets/img/credit-card/img_1.png)

From what we see here , and what we’ve gone through in our previous classes there are some fields like location, gender etc which commonly have led to biased decisions in many cases .

Apart from this, when dealing with transactions , the model has to deal with incoming streams of data and has to keep up with it and with time the streams of data can change their statistical behavior , this is known as concept drift. So the model also needs to deal with that. As an example when we talk about fraud transactions I came across an example , so names under which transactions occur can be a point of concern , they vary a lot across cultures involve some special character and in a specific case was showing increased fraud confidence for a first name with two chars. Another example can be some names using characters which are not present in the 26 english alphabets the algorithm might have issues if not trained on such characters too.

![Distribution of fraud cases by region](/assets/img/credit-card/img_2.png)

The above pie chart is an example of what % of fraud transactional data is split across countries. So a developer must take into account the composition of the data sources to make sure it’s not skewed and further in the pipeline ensure the decisions are not biased. 

And as I mentioned concept drift before , what has been observed that when we are dealing with streams of data across time , some already favored communities will continue to dominate the distribution while some disadvantaged communities may continue to do worse , and the we have to be aware that this bias that can be introduced in our pipeline at some point in the future , so instead of relying on intervention to introduce fairness a valid fairness ware classifier is maintained over the stream.Intelligibility and Fairness considerations, you would apply in every single step of the ML lifecycle to ensure responsible development and deployment of the ML system    . When we look at the graph below we see how with time the decision boundary can shift over the data and to fix this skew,we can take actions beforehand without relying on fairness intervention tasks.

![Concept Drift as it can be seen over time](/assets/img/credit-card/img_3.png)

There are a few course actions one can take to maintain fairness over the incoming stream of data called [fairness aware stream classification](https://deepai.org/publication/fairness-enhancing-interventions-in-stream-classification). But this is a step that has to be taken a little further down the pipeline in model development with continuous feedback from the data pipeline.

### Issues due to high correlation in data ###

The dataset we have for training  the model to classify transactions as fraud or non fraud is highly correlated. This correlation between the different attributes in the dataset at a certain time becomes the cause of unfairness if not handled properly. The datasets which are used to train the model for classifying debit card transactions contains the which name, race along with other relevant information of the debit card holders. During feature selection, it becomes very obvious that keeping race will definitely make the model as biased so this column has a high probability of getting removed from the feature list. But this system will still have the impact of  race because of the presence of the name column in the feature list. We know there are certain names which have a high probability of belonging to specific communities. For example, Fatima, Farhan are mainly used by muslim community as names. So even though race is not explicitly passed in the training process, the model still has it’s impact because of correlation with the name column. 
To deal with this what becomes important is not only to remove the features which directly bring biases like race but also to look for higher-order interactions between features that produce essentially the same information. 

## Testing and Deployment Process ##

Following considerations should be taken during training and deployment:

- We need to make sure that the training data matches the deployment context.
- Monitor match between training and deployment data for individual transactions.
- Gradual rollout of model using A/B testing.
- Invite stakeholders and multiple perspectives to audit the system for biases.
- We can employ a case-based-reasoning approach (CBR) for identifying similar cases of fraud where SHAP and other feature values can be used as one of the possible similarity metrics.

## Feedback Loop and Governance ##

- Monitor user complaints and reports for transactions that are incorrectly flagged.
- Identify the specific modules which need to be fixed for better predictions.
- Also, feedbacks are not representative of the transactions processed everyday for 2 main reasons : 
    - Feedbacks contain transactions that are characterized by a high probability of being frauds.
    - The proportion of frauds in feedbacks is different from the proportion of frauds occurring everyday. Thus, feedback represents a sort of biased training set.


## References ## 

- [Blog on discriminatory outcomes ](https://towardsdatascience.com/how-to-define-fairness-to-detect-and-prevent-discriminatory-outcomes-in-machine-learning-ef23fd408ef2)
- [Blog on fraud detection](https://medium.com/practical-implications-of-ml/algorithmic-fairness-in-fraud-prevention-da1ade1c4390)
- [Payments Fraud](https://www.fico.com/en/solutions/payments-fraud)
- [Publication](https://arxiv.org/abs/1907.03334)
- [Credit Card Fraud Publication](https://www.researchgate.net/publication/319867396_Credit_Card_Fraud_Detection_A_Realistic_Modeling_and_a_Novel_Learning_Strategy)
- [Fairness Enhancing INterventions](https://deepai.org/publication/fairness-enhancing-interventions-in-stream-classification)


