# Expiration Date Analyzer
Estimates expiration date of documents using word co-occurrence.

## Why I was born
Have you seen an expired tweet on Twitter? Expiration date can be used to filter such documents.

## What I can
### 1. Logging of tweets
I can log tweets with their dynamics such as retweeters, the number of retweets. To send requests more frequently, multiple tokens can be used.

### 2. Estimation of expiration date using dynamics
I can estimate expiration date of tweets from their dynamics **WITHOUT** training, so we can use the estimation as training data for 3.
$$
f(doc_i,dynamics_i(t))=\hat{t}_{expiration_i}
$$

### 3. Estimation of expiration date using word co-occurrence
Also, I can estimate expiration date of some documents using word co-occurrence. Training of a Naive Bayes classifier is required.
$$
g(doc_i)=\hat{t}_{expiration_i}
$$

### 4. To assist users to label tweets expiration date
This is critical to evaluate the models.

## Reference
Hirotaka Nagashima and Keishi Tajima. 2024. Automatic Construction of Expiration Time Expression Dataset from Retweets. In Companion Proceedings of the ACM Web Conference 2024 (WWW '24). Association for Computing Machinery, New York, NY, USA, 545–548. https://doi.org/10.1145/3589335.3651471
