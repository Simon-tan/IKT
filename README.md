# IKT



### Data format

First line : the number of skills a student attempted with student ID.
Second line : the skill id sequence.
Third line : the question id sequence.
Forth line : the response sequence.

 ```
    15,1
    1,1,1,1,7,7,9,10,10,10,10,11,11,45,54
    5001,5023,5044,5066,2014,2058,6017,20004,60001,1200,1201,1311,2014,2410,2001
    0,1,1,1,1,1,0,0,1,1,1,1,1,0,0
 ```

### Feature Engineering
1) Run FeatureEngineering.py

produce train_data.csv and test_data.csv
 



### Transform CSV to Arff (easy way to transform)
1)Copy the follwing to CSV header

 ```
@relation ASS2009
@attribute skill_ID numeric
@attribute skill_mastery numeric
@attribute ability_profile numeric
@attribute problem_difficulty numeric
@attribute correctness {1,0}
@data
```
2) change both train and test files extension from .csv to .arff



### WEKA
1) Install WEKA

 
### Run TANB classifer with WEKA
After getting the data set with features and class variables. 
Run TAN classifier under train and test setting

1) preprocess tag-> open file: training data
2) classify tag->choose: weka>classifier>bayes>BayesNet->searchAlgorithm: TAN
3) supply test set with test data
4) start to run

 
