# Ensemble-Learning-and-Tuning (Tree Pruning & Hyper Parameter Optimization)

# Pruning Basics

In DT, if there is any unnecessary edges, then we have to remove them. In the case of Neural Network, there are weight, unnecessary neural, and many layers. We have to remove those unnecessary ones. This is called 'Pruning'.

**Neural Network Pruning:**

▪ **Weight Pruning**: Removing connections with small weights or setting them to zero.

▪ **Neuron Pruning**: Removing entire neurons from the network.

This is also called **'Node Pruning'**. Removing entire node from the neural network.  

In Neural Network, there are 3 layers: Input Layer, Output Layer, and Hidden Layer. 

If there is any unnecessary hidden layer, then we can also remove those using Pruning.

▪ **Layer Pruning**: Removing entire layers of the neural network.

**Decision Tree Pruning:**

▪ **Pre-Pruning**: It involves setting a threshold on certain parameters (e.g., the maximum depth of the tree, minimum number of samples per leaf) while growing the tree. The tree construction stops when the threshold is reached, preventing the tree from growing too large. It is also a hyperparameter optimization technique.

Pruning the tree before constructing. We can create the tree based on the maximum tree depth, height, and criterion.

▪ **Post-Pruning**: This technique involves growing the tree to its full depth and then removing nodes or branches that do not provide significant predictive power. Pruning decisions are made based on metrics like cross- validation accuracy or information gain.

Pruning after constructing the tree. 

**For a big dataset, pre-pruning is generally recommended over post-pruning. Pre-pruning refers to stopping the tree construction
process early before it reaches its maximum depth, while post-pruning involves building the complete tree and then removing or
collapsing nodes to reduce overfitting.**

**Recommended**: If the dataset is small, then use Post-Pruning. if the dataset is huge, then use Pre-Pruning.


# Decision Tree Pruning


![image](https://github.com/TITHI-KHAN/Ensemble-Learning-and-Tuning/assets/65033964/ab1cd441-6c09-4c8c-b48e-34e679b28f35)

After Pruning, the tree became simple. By this, we can also predict easily. This mainly reduces overfitting.

We are going to make the tree as simple as we can by considering the prediction result. We will also have to ensure a good prediction result.


# Ensemble Learning

▪ Ensemble learning is a machine learning technique that involves combining the predictions of multiple individual models (learners) to create a more robust and accurate model. Ensemble learning means multiple model. 

▪ The idea behind ensemble learning is that by combining several weak or moderately performing models, the ensemble can achieve better overall performance and generalization compared to any single model in the ensemble.



![image](https://github.com/TITHI-KHAN/Ensemble-Learning-and-Tuning/assets/65033964/a8f17194-17e3-4f7e-afa8-114fa81e8e2f)


In Bagging, we created 4 DTs on the dataset and based on their results, the output will be the majority result.


![image](https://github.com/TITHI-KHAN/Ensemble-Learning-and-Tuning-and-Tree-Pruning-and-Hyper-Parameter-Optimization/assets/65033964/8214799f-6b07-4d60-bb0b-ff190171d8d7)


If the dataset is the same, then we will get the same tree. The model will also be the same. 

In Bagging, when each tree creates a model, then they don't use the same data. Though the dataset is single and the same, when we create a DT, then it will be created by a subset. Again, another DT will be created with another subset. Taking a few portions of the same dataset, we train the model (Ensemble). 

The Cose Complexity of DT -> There is a value of CCP-alpha. Based on that value, subset will be created. 

In Scikit-learn, **the Cost-Complexity Pruning (CCP)** is available for decision tree classifiers through the ccp_alpha parameter. CCP pruning allows you to control the complexity of the decision tree by adding a cost term for each node based on the number of samples it contains and the impurity of its leaves. By tuning the ccp_alpha parameter, you can control the amount of pruning applied to the decision tree.

**Accuracy vs Alpha graph:** 

Value of Alpha: 0.015
Training Accuracy: 0.96
Testing Accuracy: 0.94

# Ensemble Algorithms



![image](https://github.com/TITHI-KHAN/Ensemble-Learning-and-Tuning/assets/65033964/e192c203-339f-4d9a-b3a5-cb51a8690b53)




# Bagging Vs. Boosting

**Bagging (Bootstrap Aggregating):**

▪ Data Sampling Technique

▪ Model Independence 

▪ Parallel Training

▪ Aggregation Method

Bagging is a Parallel Learning.


![image](https://github.com/TITHI-KHAN/Ensemble-Learning-and-Tuning/assets/65033964/5002f894-0260-4483-867d-6ec916ae5efd)


Each sub-sample will be different, though they might contain some of the same data of the dataset. We are doing sampling here. We have separate model here. Each of the model is different. No model has any relationship with another model. Each model has been trained separately. No model has any dependency on another model. Each of them will produce a different result.


**Boosting:** 

▪ Data Weighting 

▪ Sequential Training 

▪ Model Dependence 

▪ Error-Based Weights

Boosting is a Sequential Learning.


![image](https://github.com/TITHI-KHAN/Ensemble-Learning-and-Tuning/assets/65033964/53029bc1-8a80-4f11-bc59-f42bbd41cbd5)

We have weights here. At first, it will build a model, then the error of the model will be passed down to the next step. In the next stage, it will try to correct the error. From the previous state, the next step is better. It will try to make better outcome. This works as a chain. Similarly, a model will be built. Then, error will be passed down to reduce. It will work sequentially. The result we will get from Boosting, then it will be the final result.

# Random Forest

Random forests or random decision forests is an **ensemble** learning method for classification, regression and other tasks that operates by constructing a **multitude of decision trees** at training time. For classification tasks, the output of the random forest is the class selected by **most trees**.

**Random Forest** is a classifier that contains several **decision trees** on various subsets of the given dataset and takes the **average** to improve the predictive accuracy of that dataset. Instead of relying on **one decision tree**, the random forest takes the prediction from each tree and based on the **majority votes of predictions**, and it predicts the final output. The greater number of trees in the forest leads to higher accuracy and **prevents the problem of overfitting**.



![image](https://github.com/TITHI-KHAN/Ensemble-Learning-and-Tuning/assets/65033964/45ddc77a-62a1-4188-926f-ba9f96475a0f)



**Steps involved in random forest algorithm:**

**Step 1**: In Random forest n number of random records are taken from the data set having k number of records.

**Step 2**: Individual decision trees are constructed for each sample.

**Step 3**: Each decision tree will generate an output.

**Step 4**: Final output is considered based on Majority Voting or Averaging for Classification and regression respectively.

**For example**: consider the fruit basket as the data as shown inthe figure below. Now n number of samples are taken from the fruit basket and an individual decision tree is constructed for each sample. Each decision tree will generate an output as shown in the figure. The final output is considered based on majority voting. In the below figure you can see that the majority decision tree gives output as an apple when compared to a banana, so the final output is taken as an apple.



![image](https://github.com/TITHI-KHAN/Ensemble-Learning-and-Tuning/assets/65033964/f8f62516-e2da-44a2-80ae-49b22943bc78)



![image](https://github.com/TITHI-KHAN/Ensemble-Learning-and-Tuning/assets/65033964/e8cc2672-18d9-44bf-b65f-4d53d8111704)



**Advantages and Disadvantages of Random Forest:**

▪ It reduces overfitting in decision trees and helps to improve the accuracy.

▪ It is flexible to both classification and regression problems.

▪ It works well with both categorical and continuous values

▪ It automates missing values present in the data

▪ Normalizing of data is not required as it uses a rule-based approach.

However, despite these advantages, a random forest algorithm also has some **disadvantages**:

▪ It requires much computational power as well as resources as it builds numerous trees to combine their outputs.

▪ It also requires much time for training as it combines a lot of decision trees to determine the class.

▪ Due to the ensemble of decision trees, it also suffers interpretability and fails to determine the significance of each variable.

**Applications of Random Forest:**

▪ Detect and Predict the Drug Sensitivity of a Medicine

▪ Identify a Patient’s Disease by Analyzing their Medical Records

▪ Predict Estimated Loss or Profit while Purchasing a Particular Stock

▪ Banking Industry, Credit Card Fraud Detection

▪ Customer Segmentation, Predicting Loan

▪ Healthcare and Medicine Cardiovascular Disease Prediction

▪ Diabetes Prediction, Breast Cancer Prediction

▪ Stock Market Prediction, Stock Market Sentiment Analysis, Bitcoin Price Detection, E-Commerce Product

▪ Recommendation Price, Optimization Search Ranking

# Decision Tree Vs. Random Forest



![image](https://github.com/TITHI-KHAN/Ensemble-Learning-and-Tuning/assets/65033964/79a1e12f-d1a4-4ba3-a09b-65564f408a8f)



# Hyper Param Optimization

Hyperparameter optimization, also known as hyperparameter tuning, is the process of finding the best set of hyperparameters for a machine learning model to maximize its performance on a given task. Hyperparameters are parameters that are set before the learning process begins and control aspects of the learning algorithm’s behavior, rather than being learned from the data itself.

**Grid Search**: Grid search involves defining a grid of possible hyperparameter values and evaluating the model's performance for all combinations of these values. It can be computationally expensive, but it exhaustively searches the hyperparameter space.

**Random Search**: Random search randomly samples hyperparameter values from predefined ranges. It is computationally less expensive than grid search and often provides similar or even better results.

**Bayesian Optimization**: Bayesian optimization uses probabilistic models to predict the performance of the model for different hyperparameter values and focuses on exploring regions that are likely to yield better results.

The **difference** between Grid Search and Random Search -> In Random Search, we will have to mention the no. of iteration (n_iter).

# dt_params

This likely stands for "Decision Tree parameters." In scikit-learn, Decision Trees are implemented in the `DecisionTreeClassifier` (for classification tasks) and `DecisionTreeRegressor` (for regression tasks) classes. These classes have various parameters that can be tuned to control the behavior of the decision tree, such as `criterion` (the function to measure the quality of a split), `max_depth` (the maximum depth of the tree), `min_samples_split` (the minimum number of samples required to split an internal node), etc.

#cv = 5 -> divided by 5 folds. For the stages-> T means Testing and the rest 4 places are for Training. 80% data for Training and 20% data for Testing.

![image](https://github.com/TITHI-KHAN/Ensemble-Learning-and-Tuning-and-Tree-Pruning-and-Hyper-Parameter-Optimization/assets/65033964/20df1a11-4e2c-483f-8f87-d0cd4b78d94d)


# rf_params

This is probably referring to "Random Forest parameters." Random Forest is an ensemble learning method that combines multiple decision trees to improve performance and reduce overfitting. In scikit-learn, the `RandomForestClassifier` and `RandomForestRegressor` classes are used for classification and regression tasks, respectively. Similar to decision trees, random forests have various parameters to tune, such as the number of trees (`n_estimators`), the number of features to consider for the best split (`max_features`), etc.

# Example

![image](https://github.com/TITHI-KHAN/Ensemble-Learning-and-Tuning-and-Tree-Pruning-and-Hyper-Parameter-Optimization/assets/65033964/06c46eee-b8b0-4ce0-8848-a5ec98ffa15e)

Keep in mind that the specific parameters and their default values may vary depending on the scikit-learn version. 
