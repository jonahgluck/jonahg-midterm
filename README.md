# Developing an Ensemble Boosting Model for Amazon Movie Review Classification

## Abstract

This paper presents the development of a predictive model aimed at accurately classifying Amazon Movie Reviews based on star ratings, utilizing ensemble boosting algorithms and feature engineering techniques. The challenge was to achieve high predictive accuracy without employing deep learning methods. The final model combines three boosting algorithms—**HistGradientBoostingClassifier**, **GradientBoostingClassifier**, and **XGBClassifier**—using a soft voting ensemble approach. Feature engineering involved extracting sentiment scores using TextBlob and creating a helpfulness ratio from review metadata. Additionally, attempts were made to enhance the model by identifying positive and negative keywords such as "loved," "terrible," and "excellent," but this did not significantly improve results. The model achieved an accuracy of approximately 58%. This paper details the strategy, implementation, and evaluation of the model, highlighting the effectiveness of ensemble methods and feature engineering in text classification tasks.

## Introduction

The exponential growth of user-generated content on e-commerce platforms like Amazon has made sentiment analysis an indispensable tool for businesses. Understanding customer feedback through reviews enables companies to improve products, tailor services, and enhance customer satisfaction. In this context, accurately classifying reviews based on their sentiment and content becomes crucial.

This project aimed to develop a predictive model to classify Amazon Movie Reviews into their respective star ratings (1 to 5 stars) using review metadata and textual content. The primary objectives were to achieve a high level of predictive accuracy without deep learning, ensure computational efficiency suitable for large datasets, and maintain transparency in the model's decision-making process. The challenge was to leverage ensemble-based methods, feature engineering, and boosting algorithms to create a robust model capable of handling the complexities inherent in textual data.

## Initial Approach

### Baseline Models

The initial strategy involved experimenting with baseline machine learning models commonly used in text classification.

#### K-Nearest Neighbors (KNN)

KNN is an instance-based learning algorithm that classifies new data points based on the majority class among its *k* nearest neighbors. Various distance metrics (Euclidean, Manhattan) and values of *k* were tested. The model achieved an accuracy of approximately 40%. However, it faced challenges such as high computational cost during prediction due to the need to compute distances to all training instances and ineffectiveness in high-dimensional spaces typical of textual data.

#### Naive Bayes Classifier

Naive Bayes is a probabilistic classifier based on Bayes' theorem with the assumption of feature independence. Both Multinomial and Gaussian variants were evaluated, resulting in around 40% accuracy. The primary challenge was the unrealistic independence assumption for text data, where word occurrences are often correlated.

### Limitations of Baseline Models

The unsatisfactory performance of these models highlighted their inability to capture the complex patterns and feature interactions in the dataset. This realization prompted a shift towards more sophisticated algorithms capable of handling such complexities.

## Transition to Boosting Algorithms

### Rationale for Boosting

Boosting algorithms improve predictive performance by combining multiple weak learners to form a strong learner. They are adept at handling complex relationships and are less prone to overfitting when properly tuned. Advantages of boosting include complex pattern recognition, as they can capture nonlinear relationships, improved robustness with better generalization to unseen data, and the ability to provide insights into which features contribute most to predictions. In a dataset that is skewed, as this one is, boosting algorithms seem to perform the best.

### Selection of Boosting Algorithms

The following models were selected for their complementary strengths: **HistGradientBoostingClassifier**, **GradientBoostingClassifier**, and **XGBClassifier** (XGBoost).

## Model Development

### Individual Models

#### HistGradientBoostingClassifier

**HistGradientBoostingClassifier** is an efficient implementation of gradient boosting using histogram-based binning. Its advantages include faster training on large datasets and effective handling of numerical and categorical features. Hyperparameters such as `learning_rate`, `max_iter`, and `max_depth` were tuned. Optimization involved using early stopping based on validation loss to prevent overfitting.

#### GradientBoostingClassifier

**GradientBoostingClassifier** is a traditional gradient boosting model building additive models in a forward stage-wise manner. It offers flexibility by being able to optimize various differentiable loss functions. Hyperparameters tuned included `n_estimators`, `learning_rate`, and `max_depth`. Subsampling was implemented using the `subsample` parameter to reduce overfitting.

#### XGBClassifier (XGBoost)

**XGBClassifier** is an optimized gradient boosting framework using decision trees. Its advantages are incorporating L1 and L2 regularization, supporting parallel processing, and built-in methods for handling missing data. Hyperparameters such as `eta` (learning rate), `gamma` (minimum loss reduction), `max_depth`, `lambda` (L2 regularization), and `alpha` (L1 regularization) were tuned. Cross-validation was employed to determine the optimal number of estimators (`n_estimators`).

### Ensemble Methodology

#### Soft Voting Ensemble

The three models were combined using a **VotingClassifier** with soft voting. Soft voting averages the predicted class probabilities and considers the confidence of each model. The implementation steps included training each model with optimal hyperparameters, collecting predicted probabilities from each model, averaging the probabilities, and selecting the class with the highest average probability.

#### Advantages of the Ensemble

The ensemble approach provided robustness by reducing variance and bias, mitigated individual model errors through error compensation, and improved performance by outperforming individual models.

## Feature Engineering and Extraction

Feature engineering was critical for transforming raw data into meaningful inputs for the models.

### Data Preprocessing

Missing values in `HelpfulnessNumerator` and `HelpfulnessDenominator` were filled with zeros. Data type conversion was performed to ensure correct formats, such as converting timestamps.

### Engineered Features

#### Helpfulness Score

The helpfulness score was calculated as the ratio of `HelpfulnessNumerator` to `HelpfulnessDenominator`. It was assumed that higher scores indicate more informative reviews. Edge cases were handled by setting the score to zero when `HelpfulnessDenominator` is zero.

#### Sentiment Features Using TextBlob

Polarity, ranging from -1 (negative) to 1 (positive), and subjectivity, ranging from 0 (objective) to 1 (subjective), were extracted using TextBlob applied to `reviewText`. Efficient text processing techniques were used for scalability.

#### Keyword Analysis

An attempt was made to enhance the model by identifying the presence of specific positive and negative keywords within the review text. Positive words included "loved," "excellent," and "amazing," while negative words included "terrible," "worst," and "boring." A binary feature was created indicating whether any of these keywords appeared in a review.

**Implementation:**

- Compiled lists of positive and negative keywords commonly associated with strong sentiments.
- Scanned each review for the presence of these keywords.
- Created binary features `contains_positive_keyword` and `contains_negative_keyword`.

**Results:**

Despite the intuitive appeal of this approach, incorporating these keyword features did not significantly improve the model's accuracy. The simplicity of the keyword matching failed to capture the nuances of language used in reviews.

#### Temporal Features

The `Time` feature was converted to datetime objects, and additional features such as `Year` and `Month` were extracted to capture temporal trends in reviews.

### Sentiment Analysis Optimization

#### VADER Experimentation

To reduce computational time, VADER, using `SentimentIntensityAnalyzer` from `nltk`, was experimented with. While faster, it was less accurate. The decision was made to continue with TextBlob for better accuracy.

#### Parallel Processing

Parallel processing was employed using `ThreadPoolExecutor` for concurrent processing, which reduced feature extraction time and enhanced scalability.

### Feature Selection

To avoid overfitting and reduce complexity, feature importance was analyzed and redundant features were removed. The final feature set included `HelpfulnessNumerator`, `HelpfulnessDenominator`, `Helpfulness`, `Polarity`, `Subjectivity`, `contains_positive_keyword`, `contains_negative_keyword`, `Year`, and `Month`.

## Performance Optimizations

### Hyperparameter Tuning

Hyperparameter tuning techniques such as grid search, random search, and Bayesian optimization were used. Stratified k-fold cross-validation was employed to maintain class distribution.

### Regularization in XGBoost

Regularization parameters `lambda` (L2) and `alpha` (L1) were tuned in XGBoost to prevent overfitting and improve generalization.

### Handling Class Imbalance

Class imbalance was addressed using strategies like resampling (oversampling minority classes), adjusting class weights to penalize misclassifications, and using weighted F1-score as an evaluation metric.

### Computational Efficiency

Computational efficiency was improved by enabling parallel processing in XGBoost and during sentiment analysis, leveraging multi-core processors, and managing memory usage effectively through batch processing.

## Assumptions and Observations

### Assumptions

It was assumed that higher helpfulness scores correlate with the reliability of reviews, sentiment polarity influences star ratings, and the presence of specific positive or negative keywords would enhance the model's predictive power.

### Observations

- **Polarity-Rating Correlation:** A strong alignment between sentiment polarity and star ratings was observed.
- **Keyword Feature Impact:** Incorporating positive and negative keyword features did not significantly boost model performance, suggesting that sentiment is better captured through comprehensive analysis rather than keyword presence.
- **Subjectivity Insights:** High subjectivity was linked to extreme ratings.
- **Helpfulness Patterns:** High helpfulness was associated with consensus ratings.
- **Temporal Trends:** Slight shifts in ratings over time were noted.

## Evaluation Metrics

### Accuracy

Accuracy was defined as the ratio of correct predictions over total instances but was limited as it does not account for class imbalance.

### Precision, Recall, and F1-Score

These metrics were calculated for each class to understand per-class performance.

### Confusion Matrix

The confusion matrix provided a detailed breakdown of predictions, revealing that misclassifications often occurred between adjacent ratings.

### ROC and AUC

The ROC curve and AUC were applied using a one-vs-rest approach for multi-class classification to measure the model's ability to distinguish between classes.

### Cross-Validation Scores

Stratified k-fold cross-validation showed consistent performance across folds.

## Results

The final model achieved an accuracy of approximately 58%. High precision and recall were observed for majority classes, with acceptable performance for minority classes. The inclusion of positive and negative keyword features did not yield a significant improvement in accuracy. The ensemble model outperformed individual models, demonstrating improved stability and accuracy.

### Model Comparison

The ensemble approach proved superior to individual models, highlighting the benefits of combining models for enhanced performance.

## Conclusion

The project successfully developed a predictive model for classifying Amazon Movie Reviews using ensemble boosting algorithms and feature engineering. Key findings include:

- The effectiveness of boosting algorithms in handling complex data.
- The critical importance of feature engineering for model performance.
- The limited impact of simple keyword-based features in enhancing model accuracy.
- The advantages of ensemble methods in enhancing accuracy and robustness.
- The necessity of model optimization through hyperparameter tuning and regularization.

### Future Work

Future work could involve:

- Incorporating advanced text features such as TF-IDF, word embeddings, or topic modeling to capture more nuanced linguistic patterns.
- Experimenting with deep learning techniques like neural networks for potentially higher accuracy.
- Utilizing model interpretability tools like SHAP or LIME to better understand feature contributions.
- Including reviewer history for user behavior analysis.
- Implementing automated optimization through AutoML tools.

## References

[1] Altman, N. S. (1992). An Introduction to Kernel and Nearest-Neighbor Nonparametric Regression. *The American Statistician*, 46(3), 175–185.

[2] Zhang, H. (2004). The Optimality of Naive Bayes. *AAAI*, 3(1), 562–567.

[3] Pedregosa, F., Varoquaux, G., Gramfort, A., et al. (2011). Scikit-learn: Machine Learning in Python. *Journal of Machine Learning Research*, 12, 2825–2830.

[4] Friedman, J. H. (2001). Greedy Function Approximation: A Gradient Boosting Machine. *Annals of Statistics*, 29(5), 1189–1232.

[5] Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 785–794.

[6] Kuncheva, L. I. (2004). Combining Pattern Classifiers: Methods and Algorithms. *John Wiley & Sons*.

[7] XGBoost Documentation. (2023). Parameters for Tweedie Regression. Retrieved from [https://xgboost.readthedocs.io/en/latest/parameter.html](https://xgboost.readthedocs.io/en/latest/parameter.html)

[8] Loria, S. (2018). TextBlob Documentation. Retrieved from [https://textblob.readthedocs.io/en/dev/](https://textblob.readthedocs.io/en/dev/)

[9] Hutto, C. J., & Gilbert, E. (2014). VADER: A Parsimonious Rule-Based Model for Sentiment Analysis of Social Media Text. *Proceedings of the 8th International Conference on Weblogs and Social Media (ICWSM-14)*.

[10] Sokolova, M., & Lapalme, G. (2009). A systematic analysis of performance measures for classification tasks. *Information Processing & Management*, 45(4), 427–437.

[11] Lundberg, S. M., & Lee, S.-I. (2017). A Unified Approach to Interpreting Model Predictions. *Advances in Neural Information Processing Systems*, 30.

---

*Note: This paper is prepared as part of a project to develop a predictive model for Amazon Movie Reviews classification, focusing on non-deep learning methods.*
