# Expiration Date Analyzer
Estimates expiration date of documents using word co-occurrence.

## Why I was born
Have you seen an expired tweet on Twitter? Expiration date can be used to filter such documents.

## What I can
### 1. Logging of tweets
I can log tweets with their dynamics such as retweeters, the number of retweets. To send requests more frequently, multiple tokens can be used.

### 2. Estimation of expiration date using dynamics
I can estimate expiration date of tweets from their dynamics **WITHOUT** training, so we can use the estimation as training data for 3.
<p align="center">
  <img src="https://latex.codecogs.com/gif.latex?f(doc_i,&space;dynamics_i(t))=\hat{t}_{expiration_i}">
</p>

### 3. Estimation of expiration date using word co-occurrence
Also, I can estimate expiration date of some documents using word co-occurrence. Training of a Naive Bayes classifier is required.
<p align="center">
  <img src="https://latex.codecogs.com/gif.latex?g(doc_i)=\hat{t}_{expiration_i}">
</p>

### 4. To assist users to label tweets expiration date
This is critical to evaluate the models.
