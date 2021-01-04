---
layout: post
title: Why We Need MLOps NOW
# subtitle: My thoughts on why we need MLOps now more than ever and how it ties with ML lifecyle
cover-img: /assets/img/hero-bg.jpg
thumbnail-img: /assets/img/mlops/img1.png
share-img: /assets/img/mlops/img1.png
tags: [Machine Learning,MLOps,tech]
---
I am currently reading a great book on [MLOps by Mark Treveil](https://www.oreilly.com/library/view/introducing-mlops/9781492083283/). 
The only way I have always been able to learn things have been extensive note taking and formulating my thoughts about the subject on paper. I am writing this post as a similar exercise to improve my undestanding on a bit of this fascinating topic so far.

Machine Learning Operations or MLOps is quickly becoming a extremely critical component of succesful enterprise ML project development. By defintion it is a process that helps organizations and managers to generate long term value while reducing risks associated with data science and AI initiatives. Given the degress of proliferation of ML in curren tech zeitgeist, it is still a *relatively* new concept. So why has it become suddenly so increasingly important all of a sudden ?

### MLOps vs ModelOps vs AIOps ###
Before I write further I would like to clarify the differences between these terms. 
MLOps and ModelOps are use interchageable very often. However an argument can be made that ModelOps is a more generic term compared to MLOps as the latter focuses on ML models and the former on any kind of model (eg. rule based models)

AIOps, though again similar sounding, is very different from MLOps. AIOps refers to the process of solving operatonal challenges through the use AI, in short AI for DevOps. For example creating models for anomaly detection to predict upcoming network or drive failures to alert the DevOps teams before they arrive. Working in the AIOps team at Red Hat during my internship I can attest to the differing nature of these paradigms in team strucutre and projects.

### Challenges in MLOps ###

MLOps can be summarized as the standardizing and streamlining of ML life cycles. One may ask why do we need to streamline this lifecycle ? A simple view of the lifecyle can bee seen in the picture below.

|<img src="/assets/img/mlops/img1.png" width="300" height="300">|
| :--: |
|*Source: Mehrnoosh Sameki (sameki@bu.edu)*|

We can break it down even further into : define business goal,understand and clean data,build model,deploy model & iterate.
For most ogranizations, ML models into production is a relativly new thing. Therefore until recently the smaller size of deployments and scale in general didn't require a company wide interest in managing ML lifecycles (for most orgs). But models have become extremely important , critical and in parallel need for managing model risks has also risen. Thus the reality of ML models in production is much more complex than the diagram shows us.

What are some of the challenges that complicates this cycle ? 
- Data is constantly changing. Results needs to be constantly relayed back to business stakeholders to ensure the reality of model in productions on production data continues to meet the reality of expectations and addresing original goals
- Machine Learning life cycle invovles people from business,data science,IT teams and often none of these groups are operating similar set of tools. It is also possible they do not even share the same thinking about the issue to have a good communication.
- Data Scientists are not software engineers. They specialize in model building and aren't necessarily experts in writing applications. Which means often data scientists might be juggling many roles and hence get stretched too thin. And this becomes very problematic working  at scale. Because at the end we don't want them to be vertically too siloed with this that they have to handle models they didn't create.

### What MLOps Solves ###

#### MLOps to Mitigate Risk ####

MLOps is important to any team that has even one model in production because depending on the model,performance monitoring and adjustment can be essential. By enforcing 'rules' , MLOps allows only safe and reliable actions to be taken which are instrumental in risk mitigation. 

Though to perfrom risk mitigation we have to perform risk analysis. We should be able to analyze and know :
- Risks that unavailability can produce
- Risks from bad predictions 
- Risks that model accuracy or fairness decreases over time (concept drift)

The team must ensure that risk assesment is being performed before we go on and deploy models.
So how then can we mitigate risks?.
Pushing ML models into production without MLOps infrastructure is risky for many reasons but mainly because **assessment of  model performance often can only be done in production.** That is because training data must be a good representation of data encountered in production environment and we won't know that until after the task.


ML model performance is also very sensitive to produciton environment. Becuase ML models aren't written by hand like traditional software but are machine generated. They use a lot of libararies, many of which are open-source and we can run into versioning issues.

Risk doesn't stop after succesful production deployments. Governance is a big part of ML lifecyle. As more an more models keep getting pushed , MLOps must step in to identify potential risks before they occur. Monitoring is also critical in understanding how models are being uses.

#### MLOps for Responsible AI ####

My course with AI fairness this semester at Boston University has led me to learn so much more about ML lifecycles and risks. And I can say with confidence that MLOps is absolutely imperative when it comes to AI fairness.
It is important to consider the black box nature of big deep learning models makes them lack much required transparency. It is much harder to understand predictions which in turn makes it harder for us to demonstrate compitency of our model to pass regulations. 

**Introducing automation via ML models shift the onus of accountability from bottom of hierarchy to top.** The decisions that were previously being made by humans who operated within some margin are now being made by models. The person thus reliable on model quality can be a manager for the team or maybe even an executive. This brings the need for repsonsbile AI even more to the forefront apart from the much greater social needs of course.

With MLOps we can induce the two important principles of responsible AI into the workflow: Intetionality and Accountabiltiy.
We can ensure that the model has the best intentions by making sure data comes from compliant,unbiased sources. Multiple checkpoints can be in place to chekc model bias. Intetionality also includes explainabiltiy which is a core component for all the stakeholders.

Accountability is equally important and is tied to traceability. The organization must have an overall view of how data is being used and having a centralized understanding of the model workings.


#### MLOps for Scale ####

MLOps isn't only important for the crucial task of risk assessment and mitigation; it is also an essential component to massively deploy ML models at scale. Going from a handful models to hundreds or thousands requires MLOps discipline. With good MLOps practices we can :
- Keep track of versioning, specailly during dev cycles
- Understanding if new models perform better than older ones and automaticall keeping the better ones in prod.
- Ensure that model performace is not degrading over certain time periods.

### Conclusion ###

Even with this brief discussion we can see that MLOps is not an optional practice. They are an essential part of scaling ML at an enterprise level. Teams that try to deploy models without proper MLOps practices risk quality and continuity.

I will write more about this topic soon :) 

Thanks