# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model DetailsLogistic Regression trained via scikit-learn
## Intended Use
Predict salaries over 50k USD a year based on demographic and financial data
## Training Data
Number of training data: 26048
## Evaluation Data
Number of training data: 6513
## Metrics

Calculated metrics:
precision: 0.71
recall: 0.26
fbeta: 0.38


## Ethical Considerations
Ethical considerations are dependent on the use case for this model. If we want to make use of the 
prediction on the salary regarding giving out loans or so, keep in mind that the model
makes use of demographic and ethnic data which might be inappropriate.

## Caveats and Recommendations
One has to be careful about only using categories trained by the label encoder in the training data.
If in doubt, have a closer look at the encoder saved in the model folder.
