# Accuracy scores obtained by various models:
The best accuracy scores obtained by the various models:

Logistic Regression: the best accuracy possible was obtained around 92%, while increasing the amount of features in the model reduced the accuracy to about 90%. While this appears to be an indication that some of features are not needed, the increase in accuracy is not sufficiently large to justify the removal of some of the features. This is reasoned from the fact that this dataset is not ideal in terms of size. The estimators have shown some significance in the model, and future observations may contain additional insights on the effectiveness of the variables. However, a smaller model using only the most significant variables results in

Random Forest: The random forest classifier was able to obtain accuracy between 94% and 92% by varying the number of trees in the forest. While the various trees in the forest were found to be dissimilar, it is likely that a majority were similar in structure. The discriminant focussed approach of the decision tree is considered to be highly effective for this role. However, the forest did classify 3 patients who had Parkinson's as healthy.

Support Vector Machines: The SVM classifier was again able to obtain an accuracy of about 92% and almost exacly the same as the logit model. It did however manage to only misclassify 3 observations,all three being false postitves, implying that the model did not missclassfy any patient that had Parkinson's.

ANN: The neural network using the usual ReLu activation and binary cross entropy, we get an accuracy of 97%, with a single patient who had Parkinson's being classifed as healthy.
