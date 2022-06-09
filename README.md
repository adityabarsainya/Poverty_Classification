## KDTree + XGBoost Submission - Binary Image Classification Model for Poverty Map Dataset by WILDS

### Dataset
In this project we study classification in the context of the Poverty dataset, which is part of the Wilds Project.
The goal of this project is to classify poor vs. wealthy regions in Africa based on satellite imagery. There are ~20,000 images covering 23 countries in Africa.The satellite images are of the shape 224X224X8. Each pixel corresponds to a 30mX30m area and each image corresponds to a 6.7kmX6.7km square. To see some sample images, see this notebook. 

This dataset comprises images from both urban and rural areas. In general, urban areas are significantly more wealthy than rural areas. See this notebook for details. To make the problem into a classification task, we define a threshold on the wealth that separates the poor from wealthy. As there is a large difference between rural and urban, we use a different threshold for each subset. Rural images with wealth less than -0.5 are labeled as poor and greater than -0.5 as wealthy. Similarly, we pick a threshold of 1.3 for urban images.
Dataset 

All files (train and test) are stored inside this folder in npz format. We divided this dataset into one train and 2 test sets. We separated out ~25% of the data to build a countries test set (Country_test_reduct.csv) s.t. the test countries that are not present in the training set. In the random test set, we separated 25% of the instances at random from the remaining 75% of data to generate a random test set (Random_test_reduct.csv).

Use “filename” column to get the respective sets: <br>
train.csv: Ground truth for the training dataset. Use the column “label” to train your models.<br>
Country_test_reduct.csv: Country test set. You have all the same columns as train.csv except for label and wealthpooled.<br>
Random_test_reduct.csv: Random test set. You have all the same columns as train.csv except for label and wealthpooled.<br>



### Model
The Binary Image Classification Model in this submission,  built on the base **KDTree + XGBoost** model given. Multiple changes such as data split, multiple model training, feature addition, one hot encoding, etc. were made to improve the performance of the baseline model and get a higher accuracy. The major changes has been explained as follows:

### Trained the KD Tree on higher number of images
The KDTree was trained using a higher number of images, 10000 images, from the dataset. Also, the maximum depth of the tree was increased to 10. Then the whole dataset was encoded using this trained KDTree. After encoding, we finally got an increased number of features.

### Addition of nightlight feature
The encoded data only had the information about the image but the dataframe also had additional information about the nightlight mean (given as `nl_mean`) and country labels. So, we added nightlife as a feature in the encoded data.

### One-Hot Encoding for the Countries
We used the *sklearn* library's `OneHotEncoder`, that encodes categorical features as a one-hot numeric array, to encode the Countries in our dataset. The features are encoded using a one-hot encoding scheme. This creates a binary column for each category and returns a sparse matrix or dense array.

### Split Data into Urban and Rural and run separate models
One of the major additions was splitting the data based on the `Urban` and `Rural` categories. Then, we run two separate models, *model_urban* and *model_rural*, for the two categories of data. This is done because both urban and rural images have different threshold values to determine if it's poor or wealthy. Separating them into two data sets allows the models to learn different patterns for each case, resulting in better trained models.

### Tuning the hyperparameters
The model has also been tuned by varying multiple hyperparameters to get a higher accuracy and finally select the most optimal set of hyperparameters for our model. Some of those are as follows:

##### Training split hyperparameter
For hyperparameter tuning, we split the data into a `80 (training): 20 (valid)` data split.

##### XGBClassifier Hyperparameters
We got these optimal hyperparameters for our model. Specifically, changing the following hyperparameters:

- ensemble_size: 30
- max_depth: 20
- eta: 0.03
- verbosity: 0
- objective: 'binary:logistic'
- nthread: 7
- eval_metric: ['error', 'logloss']
- num_round': 30
- gamma: 0.2
- subsample: 1
- min_child_weight: 3
- alpha: 0
- seed: 0

For each model, we ran the code for 30 iterations. Then, we kept the maximum depth of the tree as 20 for both models to prevent overfitting. Then, the learning rate was decreased to 0.03 as a higher value does not fit the model very well. The loss function to be minimized used is the binary:logistic (logistic regression for binary classification), which returns predicted probability (not class). The evaluation metrics used for the model are the negative log likelihood, and binary classification error rate. Then we tuned gamma to be 0.2 which specifies the minimum loss reduction required to make a split. For inducing regularization, we have used the alpha hyperparameter which tunes the L1 regularization. Also, we kept a seed value in order to reproduce the results.

## Training
Error and logloss graphs for the Urban and Rural models can se see here (Links has been added):
- Urban Model
[Urban Model link](https://drive.google.com/file/d/1bOSrUt7JE_mNRY0aegwB4XGbr2hnV4d7/view?usp=sharing)
- Rural Model
[Rural Model link](https://drive.google.com/file/d/1FXsyGSUgbiVRpPFK7xqx-eldu6ywvfwv/view?usp=sharing)


Once the model was hypertuned, then we trained the model on the whole dataset, removing the split. After training the model, the model was saved into a pickel file.


## Prediction
For prediciton, the testing data was divided into the two categories `urban` and `rural`. The two separate datasets were given to their respective models to predict the labels.
