# machine-learning-models
## Fuel conusmption(SLR)
We will download data by using !wget from IBM Object Storage.We have downloaded a fuel consumption dataset, FuelConsumption.csv, which contains model-specific fuel consumption ratings and estimated carbon dioxide emissions for new light-duty vehicles for retail sale in Canada.we read data in pandas dataframe. we explore data and select features engine size,cylinders,fuel consumption,co2 emmisions and create hist plots.we plot each of these plots against emision to see their relationship. we create train & test dataset and fit simple linear regression model to it.we model data using scikit learn & plot fit line over data. we evaluate model by calculating accuracy and errors. 
**Here MSE is 966.29,Mean Absolute error is 23.77,R2-score is 0.73**.

## Polynomial Regression(Co2)
we downloaded a fual consumption dataset and read into pandas dataframe.we explore data and select features engine size,cylinders,fuel consumption,co2 emmisions and create plots.we plot each of these plots against emision to see their relationship. we create train & test dataset and fit polynomial linear regression model to it by creating polynomial features through sklearn. we calculate coefficient,intercept.we model data using scikit learn & plot fit line over data. we evaluate model by calculating accuracy and errors.
Here PLR with 2 degree 
**Coefficients:  [[ 0.         28.20627576  4.65483373 -0.50961972]]
Intercept:  [130.82116603]
Mean absolute error: 21.00
Residual sum of squares (MSE): 758.80
R2-score: 0.78**
Here PLR with 3 degrees
**Coefficients:  [[ 0.         28.20627576  4.65483373 -0.50961972]]
Intercept:  [130.82116603]
Mean absolute error: 21.00
Residual sum of squares (MSE): 758.80
R2-score: 0.78**

## Multiple linear regression(Co2)
In this, we take Fuel consumption.csv dataset. we take engine size, fuel consumption,cylinders,co2 emmission, fuel consumption city,fuel consumption hwy as one dataframe as clf.we split data into training and testing datasets. we created multiple linear regression model for co2 emmision as dependent variable and others as independent variables. we find coefficients,residual sum of squares, variance square usig scikit libraries. we will take decision by considering varianve score and residual sum of squares.
**Coefficients:  [[11.38129021  7.5352014   5.2041267   4.04496166]]
Residual sum of squares: 469.11
Variance score: 0.88**

## China GDP(Non-lr)
In this  we take china gdp dataset, it contains years from 1960 to 2014 and its GDP values. we plot graph against years and values it is not linear and curvy, It looks like expo so we take sigmoid model use curve_fit to create it and fit it.we have to find beta1, beta2 . we then split data into test and train find MSE R2 score to check accuracy of model.
**mean absolute error: 0.025173
Resiudal sum of squares(MSE): 0.001054
R2-score: 0.98**

## K-Nearest neighbors Cust cat
In this we take telecustomers dataset in whick customers are in 4 categogeris so we use classifier and KNN method. we load data into pandas datafreame.we normalize it into float type and split data into train and test data with 80 & 20%. First we take k=4 train model and predict it and find accuracy of it by train and split classifier method.then we take k= 6 and again train predict accuracy. Now we have to check which one is best fit for choosing k. so we calcute accuracy of all ks 1-10 and plot graph . In plot we can say best value of k is with high accuracy. here k=9 gives high accuracy. so best model comes for a value of k=9.
**The best accuracy was with 0.34 with k= 9**

## Decision Tress Drug
In this we take drug dataset and convert into pandas and normalize data by lebeling dummy variables to columns,sex,BP,Cholestrol. we then split data into trainset and testset. we create model as drugtree and predict it as predtree and find metrics accuracy score. to visualize we need some packages we need to install .import libraries and plot decision tree.

## Logistic Regression- Churn Prediction
In this, we take telecommunication data. first we download data from IBM cloud storage and convert into pandas dataframe. we choose few columns as X are independent variables and y churn prediction as dependent variable.we preprocess data as converting churn into integer. we split data into train and test sets.we build logistic regression model as bilenear as solver predict test set. now we evaluate with confusion matrix. we find precision,recall,accuracy by classification report.we calculate logloss. we can build model using onother solver sag and calculate logloss.logloss of liblinear is 0.60 and **logloss of sag is 0.61**.so probability of performance of sag model is higher than probability of performance of bilinear model.

## Support Vector Machines(SVM)-Classify human cells
we take cancer dataset that is publicly available from the UCI Machine Learning Repository (Asuncion and Newman, 2007)[http://mlearn.ics.uci.edu/MLRepository.html]. The dataset consists of several hundred human cell sample records, each of which contains the values of a set of cell characteristics.we download from IBM cloud storage. we convert it into pandas dataframe. we preprocess data. we have to find whether the cell is malign or not labelled as 2 & 4. we convert class which is Y dependent variable into integer. we take'Clump', 'UnifSize', 'UnifShape', 'MargAdh', 'SingEpiSize', 'BareNuc', 'BlandChrom', 'NormNucl', 'Mit' as X. we split data into train and test dataset. we train model by SVM method RBF and predict it .To evaluate model we calculate confusion matrix to find precision,accuracy,recall.we calculate jaccard score for accuracy. And we again create model by linear method predict test set. we calculte jaccard score for accuracy of the same. Here both jaccard scores are same 0.944. So we can build models by RBF and LINEAR both are good fit.

## K-means Clustering-customer segmentation
we download cust segmentation dataset and stored in pandas dataframe.we preprocess data by removing address column.normalize data using standaard scalar.we apply k-means on our dataset.We assign the labels to each row in the dataframe.We can easily check the centroid values by averaging the features in each cluster.Now, let's look at the distribution of customers based on their age and income.k-means will partition your customers into mutually exclusive groups, for example, into 3 clusters. The customers in each cluster are similar to each other demographically. Now we can create a profile for each group, considering the common characteristics of each cluster. For example, the 3 clusters can be:
- AFFLUENT, EDUCATED AND OLD AGED.
- MIDDLE AGED AND MIDDLE INCOME.
- YOUNG AND LOW INCOME.

## Hierarchial clustering-cars
A Hierarchical clustering is typically visualized as a dendrogram as shown in the following cell. Each merge is represented by a horizontal line. The y-coordinate of the horizontal line is the similarity of the two clusters that were merged, where cities are viewed as singleton clusters. By moving up from the bottom layer to the top node, a dendrogram allows us to reconstruct the history of merges that resulted in the depicted clustering.we cleaned data of vehicle dataset. we clustered using scipy and scikit-learn.
