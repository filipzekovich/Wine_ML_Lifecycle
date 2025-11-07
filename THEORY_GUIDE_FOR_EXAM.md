# COMPLETE MACHINE LEARNING THEORY GUIDE
## For Total Beginners - Everything You Need for Your Exam

---

## TABLE OF CONTENTS
1. [What is Machine Learning?](#1-what-is-machine-learning)
2. [Types of Machine Learning](#2-types-of-machine-learning)
3. [The Machine Learning Workflow](#3-the-machine-learning-workflow)
4. [Data Concepts](#4-data-concepts)
5. [Data Preprocessing](#5-data-preprocessing)
6. [Feature Engineering](#6-feature-engineering)
7. [Classification vs Regression](#7-classification-vs-regression)
8. [Model Training](#8-model-training)
9. [Model Evaluation](#9-model-evaluation)
10. [Our Specific Models](#10-our-specific-models)
11. [Hyperparameter Tuning](#11-hyperparameter-tuning)
12. [Deployment](#12-deployment)
13. [Common Terms Dictionary](#13-common-terms-dictionary)

---

## 1. WHAT IS MACHINE LEARNING?

### Simple Definition
**Machine Learning = Teaching computers to learn from examples instead of programming explicit rules**

### Traditional Programming vs Machine Learning

**Traditional Programming:**
```
IF alcohol > 12 AND volatile_acidity < 0.5 THEN
    wine_quality = "Good"
ELSE
    wine_quality = "Not Good"
```
Problem: We have to write rules manually. What if there are 100 features? Impossible!

**Machine Learning:**
```
Give computer 1000 examples of wines with their quality
Computer finds patterns automatically
Computer can predict quality of NEW wines
```

### Real World Analogy
**Learning to recognize dogs:**
- Traditional: Write rules (has 4 legs, has fur, barks, etc.)
- Machine Learning: Show 1000 pictures of dogs and cats, computer learns what makes a dog

---

## 2. TYPES OF MACHINE LEARNING

### Supervised Learning ⭐ (What we use)
- **You give**: Input features (wine properties) + Output labels (quality)
- **Computer learns**: Pattern from inputs to outputs
- **Goal**: Predict output for NEW inputs
- **Example**: Predict wine quality from chemical properties

### Unsupervised Learning
- **You give**: Only input features (no labels)
- **Computer finds**: Hidden patterns or groups
- **Example**: Group similar wines together (clustering)

### Reinforcement Learning
- **Computer learns**: By trial and error with rewards
- **Example**: Teaching robot to walk, game AI

**Our project uses SUPERVISED LEARNING**

---

## 3. THE MACHINE LEARNING WORKFLOW

```
Raw Data
    ↓
1. DATA UNDERSTANDING
   (Explore, visualize, find patterns)
    ↓
2. DATA PREPARATION
   (Clean, transform, create features)
    ↓
3. SPLIT DATA
   (80% training, 20% testing)
    ↓
4. PREPROCESSING
   (Standardize, encode, reduce dimensions)
    ↓
5. MODEL TRAINING
   (Teach model on training data)
    ↓
6. MODEL EVALUATION
   (Test on unseen test data)
    ↓
7. HYPERPARAMETER TUNING
   (Optimize model settings)
    ↓
8. DEPLOYMENT
   (Put model in production API)
```

---

## 4. DATA CONCEPTS

### Features (X) - Independent Variables
**Definition**: Input variables used to make predictions

**In our project (12 features):**
- fixed_acidity
- volatile_acidity
- citric_acid
- residual_sugar
- chlorides
- free_sulfur_dioxide
- total_sulfur_dioxide
- density
- pH
- sulphates
- alcohol
- color

**Analogy**: Like ingredients in a recipe

### Target (y) - Dependent Variable
**Definition**: The thing we want to predict

**In our project:**
- Original: quality (0-10)
- Transformed: quality_label (0 or 1)
  - 0 = Not Good (quality < 7)
  - 1 = Good (quality ≥ 7)

**Analogy**: Like the final dish's rating

### Dataset
**Definition**: Collection of examples used for training/testing

**Our dataset:**
- 6,497 wines total
- 1,599 red + 4,898 white
- 13 columns (12 features + 1 target)

---

## 5. DATA PREPROCESSING

### Why Preprocess?
**Problem**: Raw data has issues that confuse machine learning models
- Missing values (gaps in data)
- Outliers (extreme unusual values)
- Different scales (alcohol is 8-15, pH is 2-4)
- Different types (numbers vs categories)

**Solution**: Clean and transform data before training

### 5.1 Handling Missing Values

**What are missing values?**
- Data points that weren't recorded
- Shown as NaN (Not a Number) in Python

**Example:**
```
Wine 1: alcohol = 12.5
Wine 2: alcohol = NaN    ← Missing!
Wine 3: alcohol = 11.2
```

**Solutions:**
1. **Delete rows** - Remove wines with missing data (loses data!)
2. **Fill with mean** - Replace NaN with average (affected by outliers)
3. **Fill with median** ⭐ (What we use) - Replace NaN with middle value (robust)

**Our approach:**
```python
df[column].fillna(df[column].median())
```

### 5.2 Handling Outliers

**What are outliers?**
- Values that are extremely different from others
- Can be errors OR natural variation

**Example:**
```
Normal alcohol values: [9.4, 9.8, 10.2, 10.5, 11.1]
Outlier: [45.7]  ← Unrealistic!
```

**Why remove?**
- Outliers can mislead the model
- Model might think 45% alcohol is normal

**Our approach - Clipping:**
```
Keep values between 1st percentile and 99th percentile
Values below 1st → become 1st percentile value
Values above 99th → become 99th percentile value
```

**Example:**
```
Original: [2, 5, 8, 10, 12, 50, 100]
1st percentile: 2
99th percentile: 50
After clipping: [2, 5, 8, 10, 12, 50, 50]
```

### 5.3 Standardization (Scaling)

**Problem**: Features have different ranges
```
alcohol: 8 to 15 (range of 7)
pH: 2.7 to 4.0 (range of 1.3)
total_sulfur_dioxide: 6 to 440 (range of 434)
```

**Why this matters:**
- Models treat large numbers as more important
- Sulfur dioxide would dominate just because numbers are bigger!

**Solution - StandardScaler:**
Transform all features to have:
- Mean (average) = 0
- Standard Deviation = 1

**Formula:**
```
scaled_value = (original_value - mean) / standard_deviation
```

**Example:**
```
Original alcohol values: [8, 10, 12, 14]
Mean = 11, Std = 2.45
After scaling: [-1.22, -0.41, 0.41, 1.22]
```

Now all features are on the same scale!

### 5.4 Log Transformation

**Problem**: Some features are skewed (not normally distributed)
```
Most wines have residual sugar around 2-5
Few wines have residual sugar of 60+ (very sweet dessert wines)
Distribution is stretched to the right
```

**Solution**: Apply logarithm to compress large values
```python
np.log1p(x)  # log(1 + x)
```

**Why log1p instead of log?**
- log(0) = undefined (error!)
- log1p(0) = 0 (works!)

**Effect:**
```
Original: [1, 10, 100, 1000]
After log: [0.69, 2.40, 4.62, 6.91]
```
Values are more evenly spaced!

### 5.5 One-Hot Encoding

**Problem**: Machine learning models only understand numbers, not text

**What we have:**
```
Wine 1: color = "red"
Wine 2: color = "white"
```

**Naive approach (WRONG):**
```
red = 0, white = 1
```
Problem: Model thinks white (1) > red (0), but there's no order!

**Correct approach - One-Hot Encoding:**
```
Create binary columns:
Wine 1: red=1, white=0
Wine 2: red=0, white=1
```

**Our approach (with drop="first"):**
```
Only create ONE column: white (0 or 1)
If white=0 → it's red
If white=1 → it's white
```
This prevents redundancy (multicollinearity)

---

## 6. FEATURE ENGINEERING

### What is it?
**Creating NEW features from existing ones to help the model learn better**

### Our Feature: sulfur_ratio

**Created from:**
```python
sulfur_ratio = free_sulfur_dioxide / total_sulfur_dioxide
```

**Why this helps:**
- Original: Two separate numbers (free SO2, total SO2)
- New: One ratio showing proportion of active preservative
- Model can learn: "wines with 30% free sulfur ratio are better"

**Analogy:**
- Original: "Car travels 100 miles, takes 2 hours"
- Engineered: "Car speed = 50 mph"
- The ratio captures a meaningful relationship!

---

## 7. CLASSIFICATION VS REGRESSION

### Regression
**Predict a CONTINUOUS number**
- Examples: price, temperature, age
- Output can be any number: 5.3, 7.8, 10.2

### Classification ⭐ (What we use)
**Predict a CATEGORY/CLASS**
- Examples: spam/not spam, dog/cat, good/bad
- Output is a discrete label: 0 or 1, "red" or "white"

### Our Project Transformation

**Started as Regression:**
- Predict quality: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10
- 11 possible outputs

**Transformed to Binary Classification:**
- Predict quality_label: 0 or 1
- 0 = Not Good (quality < 7)
- 1 = Good (quality ≥ 7)
- Only 2 possible outputs

**Why transform?**
1. **More practical**: Wine producers want pass/fail, not exact score
2. **Easier to evaluate**: Classification metrics are clearer
3. **Better performance**: Binary classification is simpler to learn

---

## 8. MODEL TRAINING

### What is Training?

**Simple explanation:**
The model looks at many examples and learns patterns.

**Process:**
```
1. Model starts with random guesses
2. Makes prediction on training data
3. Checks how wrong it is (error)
4. Adjusts internal parameters to reduce error
5. Repeat steps 2-4 thousands of times
6. Eventually learns good patterns
```

**Analogy:**
Like a student learning math:
- See many practice problems (training data)
- Try to solve them
- Check answers
- Learn from mistakes
- Eventually can solve new problems (test data)

### Train-Test Split

**Why split data?**
We need to test the model on DATA IT HAS NEVER SEEN!

**Our split:**
- **Training set (80%)**: 5,197 wines
  - Model learns from these
  - Sees correct answers

- **Test set (20%)**: 1,300 wines
  - Model has never seen these
  - We check predictions to see if model generalized

**Analogy:**
- Training set = practice problems with answers
- Test set = final exam (no answers, test what you learned)

### Why Stratified Split?

**Problem**: Our classes are imbalanced
- 80% Not Good wines
- 20% Good wines

**Solution - Stratified Split:**
Keep the SAME ratio in both train and test
- Training: 80% Not Good, 20% Good
- Test: 80% Not Good, 20% Good

**Why this matters:**
If test set had 50% good wines (by random chance), it wouldn't represent real-world data!

---

## 9. MODEL EVALUATION

### Why Not Use Accuracy?

**Accuracy = (Correct Predictions) / (Total Predictions)**

**Problem with imbalanced data:**
```
Dataset: 800 Not Good, 200 Good wines

Dumb Model: Always predict "Not Good"
Accuracy = 800/1000 = 80% ← Looks good!
But the model is useless (never predicts Good wines)
```

### Better Metrics for Classification

#### Confusion Matrix

```
                    Predicted
                Not Good   Good
Actual  Not Good    TN      FP
        Good        FN      TP
```

**Four outcomes:**
- **True Negative (TN)**: Correctly predicted Not Good
- **False Positive (FP)**: Predicted Good, actually Not Good (Type I error)
- **False Negative (FN)**: Predicted Not Good, actually Good (Type II error)
- **True Positive (TP)**: Correctly predicted Good

**Example:**
```
1000 wines predicted:
TN = 650 (correctly found 650 not good wines)
FP = 150 (wrongly called 150 bad wines good)
FN = 50 (missed 50 good wines)
TP = 150 (correctly found 150 good wines)
```

#### Precision

**Precision = TP / (TP + FP)**

**Question it answers**: "Of all wines I predicted as Good, how many were actually Good?"

**Example:**
```
Predicted 300 wines as Good
200 were actually Good (TP)
100 were actually Not Good (FP)
Precision = 200/300 = 0.67 (67%)
```

**When to care**: When false positives are costly
- Example: Email spam filter (don't want to mark important emails as spam)

#### Recall (Sensitivity)

**Recall = TP / (TP + FN)**

**Question it answers**: "Of all actually Good wines, how many did I find?"

**Example:**
```
250 wines are actually Good
200 were predicted as Good (TP)
50 were missed (FN)
Recall = 200/250 = 0.80 (80%)
```

**When to care**: When false negatives are costly
- Example: Cancer detection (don't want to miss any cancer cases)

#### F1 Score ⭐ (What we use as PRIMARY metric)

**F1 = 2 × (Precision × Recall) / (Precision + Recall)**

**What it is**: Harmonic mean of Precision and Recall

**Why use it**:
- Balances both precision and recall
- Good for imbalanced datasets
- One number that captures overall performance

**Example:**
```
Precision = 0.67
Recall = 0.80
F1 = 2 × (0.67 × 0.80) / (0.67 + 0.80) = 0.729
```

**Interpretation:**
- F1 = 1.0 → Perfect
- F1 = 0.729 → Pretty good
- F1 = 0.5 → Mediocre
- F1 = 0.0 → Terrible

#### ROC-AUC ⭐ (What we use as SECONDARY metric)

**ROC Curve**: Plot of True Positive Rate vs False Positive Rate

**AUC (Area Under Curve)**: Number from 0 to 1

**What it measures**: How well the model separates classes

**Interpretation:**
- AUC = 1.0 → Perfect separation
- AUC = 0.9 → Excellent (our model!)
- AUC = 0.8 → Good
- AUC = 0.5 → Random guessing (coin flip)
- AUC = 0.0 → Always wrong

**Intuitive meaning:**
"If I pick one random Good wine and one random Not Good wine, what's the probability the model ranks the Good wine higher?"

**Our model: ROC-AUC = 0.905**
→ 90.5% chance it correctly ranks a good wine above a bad wine!

### Cross-Validation

**Problem**: One train/test split might be lucky or unlucky

**Solution - K-Fold Cross-Validation:**
```
Split data into K parts (we use K=5)

Round 1: Train on parts 1,2,3,4 → Test on part 5
Round 2: Train on parts 1,2,3,5 → Test on part 4
Round 3: Train on parts 1,2,4,5 → Test on part 3
Round 4: Train on parts 1,3,4,5 → Test on part 2
Round 5: Train on parts 2,3,4,5 → Test on part 1

Average the 5 results → Reliable performance estimate
```

**Why we use Stratified K-Fold:**
Each fold keeps the same 80/20 class ratio

---

## 10. OUR SPECIFIC MODELS

### 10.1 Logistic Regression (Baseline)

**Type**: Linear model

**How it works:**
```
1. Combines features with weights:
   score = w1×alcohol + w2×volatile_acidity + ... + bias

2. Applies sigmoid function to get probability:
   probability = 1 / (1 + e^(-score))

3. If probability > 0.5 → predict Good
   If probability < 0.5 → predict Not Good
```

**Strengths:**
- Fast to train
- Interpretable (can see feature weights)
- Works well for linearly separable data

**Weaknesses:**
- Can't capture complex patterns
- Assumes linear relationship

**Our results:**
- F1: 0.525 (worst of all models)
- ROC-AUC: 0.802

**Why it's the baseline**: Simple model to compare against

### 10.2 Random Forest ⭐ (OUR WINNER)

**Type**: Ensemble of decision trees

**How it works:**

**Step 1 - Decision Tree (single tree):**
```
                  alcohol < 10.5?
                 /              \
              Yes                No
             /                    \
    volatile_acidity < 0.5?     Predict Good
       /              \
     Yes              No
     /                 \
Predict Good      Predict Not Good
```

A tree asks questions about features and makes decisions

**Step 2 - Random Forest (many trees):**
```
Build 200 different decision trees
Each tree trained on:
- Random subset of data
- Random subset of features

Final prediction = Majority vote of all trees
If 120/200 trees say "Good" → Predict Good
```

**Why "Random"?**
- Each tree sees different data (bootstrap sampling)
- Each tree uses different features
- This creates diverse trees that make different mistakes
- Averaging cancels out individual errors

**Strengths:**
- Handles non-linear relationships
- Robust to outliers
- No feature scaling needed
- Works with mixed data types
- Gives feature importance

**Weaknesses:**
- Slower to train than linear models
- Harder to interpret (black box)
- Can overfit with too many deep trees

**Hyperparameters:**
- `n_estimators=200`: Build 200 trees (more = better but slower)
- `max_depth=None`: Let trees grow fully (captures complex patterns)
- `min_samples_split=10`: Need 10 samples to split node (prevents overfitting)
- `min_samples_leaf=4`: Each leaf needs 4 samples (prevents overfitting)
- `max_features='sqrt'`: Use √(n_features) at each split (increases diversity)
- `class_weight='balanced'`: Handle imbalanced classes

**Our results:**
- F1: 0.665 (BEST!)
- ROC-AUC: 0.905 (BEST!)

**Why it won:**
- Best balance of precision and recall
- Robust to our data quality issues
- Handles feature interactions well

### 10.3 Gradient Boosting

**Type**: Ensemble of decision trees (sequential)

**How it works:**
```
1. Build first tree
2. Find mistakes (residual errors)
3. Build second tree to correct those mistakes
4. Build third tree to correct remaining mistakes
5. Repeat 200 times
6. Final prediction = Sum of all tree predictions
```

**Key difference from Random Forest:**
- Random Forest: Trees are independent (parallel)
- Gradient Boosting: Trees learn from previous trees' mistakes (sequential)

**Strengths:**
- Often highest accuracy
- Good for complex patterns
- Handles interactions well

**Weaknesses:**
- More prone to overfitting
- Slower to train (can't parallelize)
- Sensitive to hyperparameters

**Hyperparameters:**
- `n_estimators=200`: Number of trees
- `learning_rate=0.1`: How much each tree contributes (smaller = more robust)
- `max_depth=7`: Limit tree depth (prevent overfitting)

**Our results:**
- F1: 0.621
- ROC-AUC: 0.894

**Why it didn't win:**
- Lower F1 than Random Forest
- More complex without better results
- No class_weight option (harder to handle imbalance)

### 10.4 Support Vector Classifier (SVC)

**Type**: Kernel-based classifier

**How it works:**
```
1. Find a hyperplane (decision boundary) that separates classes
2. Maximize the margin (distance) between classes
3. Use RBF kernel to handle non-linear separation
```

**RBF Kernel:**
- Transforms data to higher dimension
- Makes non-linear patterns linearly separable
- Like lifting a tangled rope to untangle it

**Strengths:**
- Effective in high dimensions
- Memory efficient
- Works well when classes are well-separated

**Weaknesses:**
- Slow with large datasets
- Needs careful hyperparameter tuning
- Hard to interpret

**Hyperparameters:**
- `C=1`: Regularization (balance margin vs errors)
- `gamma=1`: Kernel coefficient (influence of single point)

**Our results:**
- F1: 0.613
- ROC-AUC: 0.889

**Why it didn't win:**
- Good performance but not the best
- More computationally expensive
- No clear advantage over Random Forest

---

## 11. HYPERPARAMETER TUNING

### What are Hyperparameters?

**Parameters**: Learned during training (e.g., tree splits, weights)

**Hyperparameters**: Set BEFORE training (e.g., number of trees, learning rate)

**Analogy:**
- Hyperparameters = Settings on the washing machine (temperature, cycle time)
- Parameters = What the machine learns about the specific load

### Why Tune?

**Default settings might not be optimal for your specific data**

**Example:**
```
Random Forest with 10 trees (default): F1 = 0.55
Random Forest with 200 trees (tuned): F1 = 0.65
```

### RandomizedSearchCV (What we use)

**How it works:**
```
1. Define hyperparameter ranges:
   n_estimators: [100, 200, 500]
   max_depth: [None, 5, 10, 20]
   ...

2. Randomly try 20 combinations
3. For each combination:
   - Do 5-fold cross-validation
   - Calculate F1 score
4. Pick the best combination
```

**Why randomized instead of grid?**
- Grid: Try ALL combinations (very slow)
- Randomized: Try random subset (faster, good enough)

**Our tuning:**
- 20 iterations
- 5-fold CV = 100 total model trains (20 × 5)
- Best parameters found automatically

---

## 12. DEPLOYMENT

### What is Deployment?

**Making the model available for real-world use**

**Not deployed:**
- Model sits on your laptop
- Only you can use it
- Have to run Python scripts manually

**Deployed:**
- Model running on a server
- Anyone can use it via internet
- Automatic predictions via API

### API (Application Programming Interface)

**What is an API?**
A way for programs to talk to each other over the internet

**Analogy**: Restaurant
- You (client) → Order food
- Kitchen (API) → Prepares food
- You receive → Food (response)

### REST API

**REST = Representational State Transfer**

**Key concepts:**
- **Endpoints**: URLs that do specific things
  - `/predict` - Make prediction
  - `/health` - Check status

- **HTTP Methods**:
  - GET - Retrieve information (like reading)
  - POST - Send information (like submitting form)

- **Request/Response**:
  - Request: Send wine features as JSON
  - Response: Get prediction as JSON

### Our API Endpoints

**1. GET /** (Homepage)
```
Request: Visit http://localhost:8000/
Response: {"status": "active", "endpoints": {...}}
```

**2. GET /health** (Health Check)
```
Request: GET http://localhost:8000/health
Response: {"status": "healthy", "model_loaded": true}
```

**3. POST /predict** (Single Prediction)
```
Request: Send wine features
{
  "fixed_acidity": 7.4,
  "alcohol": 9.4,
  ...
}

Response: Get prediction
{
  "prediction": 0,
  "prediction_label": "Not Good Quality (<7)",
  "probability": 0.8523
}
```

**4. POST /predict/batch** (Multiple Predictions)
```
Request: Send array of wines
[{wine1}, {wine2}, {wine3}]

Response: Get array of predictions
{
  "predictions": [{...}, {...}, {...}],
  "count": 3
}
```

### FastAPI

**Why FastAPI?**
- Modern Python web framework
- Automatic data validation (Pydantic)
- Auto-generated documentation (/docs)
- Fast and easy to use

### Swagger UI

**What is it?**
Interactive API documentation at `/docs`

**What you can do:**
- See all endpoints
- Read descriptions
- Test endpoints directly in browser
- No coding needed!

---

## 13. COMMON TERMS DICTIONARY

### A

**Accuracy**: (TP + TN) / (Total) - Percentage of correct predictions

**Algorithm**: Step-by-step procedure for solving a problem

**API**: Application Programming Interface - Way for programs to communicate

### B

**Bias**: Systematic error in predictions (underfitting)

**Binary Classification**: Predicting one of two classes (0 or 1)

**Bootstrap**: Random sampling with replacement

### C

**Classification**: Predicting categories/classes

**Class Imbalance**: When one class has many more examples than another

**Cross-Validation**: Testing model on multiple train/test splits

### D

**DataFrame**: Table structure for data (pandas)

**Dataset**: Collection of data examples

**Deployment**: Making model available in production

### E

**Ensemble**: Combining multiple models for better predictions

**Epoch**: One pass through entire training data

**Evaluation**: Measuring model performance

### F

**Feature**: Input variable used for prediction

**Feature Engineering**: Creating new features from existing ones

**F1 Score**: Harmonic mean of precision and recall

### G

**Generalization**: Model's ability to perform well on unseen data

### H

**Hyperparameter**: Setting configured before training

**HTTP**: Protocol for web communication

### I

**Imbalanced Data**: Unequal distribution of classes

### J

**JSON**: JavaScript Object Notation - Data format for APIs

### K

**K-Fold**: Splitting data into K parts for cross-validation

### L

**Label**: Target value/class we want to predict

**Loss Function**: Measure of model error

### M

**Machine Learning**: Computer learning patterns from data

**Median**: Middle value when data is sorted

**Model**: Trained algorithm that makes predictions

### N

**NaN**: Not a Number - Missing value

**Normalization**: Scaling features to a standard range

### O

**One-Hot Encoding**: Converting categories to binary columns

**Outlier**: Extreme value far from others

**Overfitting**: Model memorizes training data, fails on test data

### P

**Parameter**: Value learned during training

**Pipeline**: Chain of data processing steps

**Precision**: TP / (TP + FP) - Accuracy of positive predictions

**Preprocessing**: Cleaning and transforming raw data

**Probability**: Likelihood of an event (0 to 1)

### Q

**Quality**: In our project, wine rating (0-10)

### R

**Random Forest**: Ensemble of decision trees

**Recall**: TP / (TP + FN) - Coverage of actual positives

**Regression**: Predicting continuous numbers

**ROC-AUC**: Area under ROC curve - Model discrimination ability

### S

**Scikit-learn**: Python library for machine learning

**Standardization**: Scaling to mean=0, std=1

**Stratified**: Maintaining class proportions in splits

**Supervised Learning**: Learning from labeled examples

### T

**Target**: Variable we want to predict

**Test Set**: Data for evaluating trained model

**Training**: Process of teaching model from data

**Train Set**: Data for teaching the model

### U

**Underfitting**: Model too simple to capture patterns

**Unsupervised Learning**: Learning from unlabeled data

### V

**Validation**: Checking model performance

**Variance**: Model sensitivity to training data fluctuations

### W

**Weight**: Importance value for features (learned parameter)

### X

**X**: Convention for features/inputs

### Y

**y**: Convention for target/output

---

## QUICK REFERENCE CHEAT SHEET

### Our Project Summary

**Problem**: Predict if wine is Good (≥7) or Not Good (<7)

**Data**: 6,497 wines, 12 features

**Preprocessing**:
1. Fill missing → median
2. Clip outliers → 1st-99th percentile
3. Engineer feature → sulfur_ratio
4. Log transform → np.log1p
5. Standardize → mean=0, std=1
6. PCA → keep 95% variance
7. One-hot encode → color

**Models Tested**:
1. Logistic Regression (baseline)
2. Random Forest (WINNER)
3. Gradient Boosting
4. SVC

**Best Model**: Random Forest
- F1: 0.665
- ROC-AUC: 0.905
- n_estimators: 200
- max_depth: None
- min_samples_split: 10
- min_samples_leaf: 4
- max_features: 'sqrt'

**Deployment**: FastAPI REST API
- Endpoint: POST /predict
- Input: Wine features (JSON)
- Output: Prediction + confidence

---

## FINAL EXAM TIPS

### What to Focus On

1. **Understand the workflow**: Data → Preprocessing → Training → Evaluation → Deployment

2. **Know why we made each choice**:
   - Binary classification → More practical than regression
   - F1 score → Best for imbalanced data
   - Random Forest → Best balance of performance and robustness

3. **Be able to explain in simple terms**:
   - What is machine learning?
   - What does preprocessing do?
   - Why do we need train/test split?
   - What makes a good model?

4. **Know the numbers**:
   - 6,497 wines (1,599 red + 4,898 white)
   - 80/20 train/test split
   - F1: 0.665, ROC-AUC: 0.905
   - 200 trees in Random Forest

5. **Understand the code comments**: Read through all the commented code files!

### Common Questions & Answers

**Q: Why did you choose Random Forest?**
A: Best F1 score (0.665) and ROC-AUC (0.905), robust to outliers, handles non-linear relationships, and provides feature importance.

**Q: Why not use accuracy?**
A: Dataset is imbalanced (80/20). A model always predicting "Not Good" would get 80% accuracy but be useless. F1 score better captures performance on both classes.

**Q: What is the API for?**
A: It deploys the model so users can get predictions by sending wine features over the internet, without needing Python or the model on their computer.

**Q: What does preprocessing do?**
A: Cleans data (handle missing/outliers), transforms features (log, standardize), reduces dimensions (PCA), and encodes categories (one-hot) so the model can learn effectively.

**Q: How do you know the model works?**
A: Tested on unseen test data (1,300 wines the model never saw during training) and achieved high F1 (0.665) and ROC-AUC (0.905) scores.

---

**Good luck on your exam! You've got this!** 🍷🎓
