# CNN Feature Maps Visualization for Emotions using FER 2013 Dataset
This project aims to visualize the feature maps of a Convolutional Neural Network (CNN) trained on the FER 2013 dataset to predict emotions. The FER 2013 dataset contains facial images labeled with various emotions, including angry, happy, sad, and more.

## Dataset
The FER 2013 dataset used in this project can be obtained from the following link: <br>[FER 2013 Dataset](https://www.kaggle.com/datasets/msambare/fer2013)<br>

The dataset consists of grayscale images of size 48x48 pixels, and each image is labeled with one of the following emotions:<br>

Angry<br>
Disgust<br>
Fear<br>
Happy<br>
Sad<br>
Surprise<br>
Neutral<br>
For this project, we will focus on the emotions of angry, happy, and sad.

## Data Preprocessing
Before visualizing the feature maps, it's necessary to preprocess the dataset. This step involves loading and resizing the images, converting them to grayscale, normalizing the pixel values, and splitting the dataset into training and testing sets.

## Model Architecture
The CNN model used for this project will have multiple convolutional and pooling layers followed by fully connected layers. The specific architecture details and hyperparameters can be found in the accompanying notebook or script.

## Model Training
Train the CNN model using the preprocessed dataset. Use appropriate training techniques such as batch normalization, dropout, and optimization algorithms like Adam to improve the model's performance.

## Feature Maps
To visualize the feature maps of the trained CNN model, we can follow the following steps:<br><br>
Load the pre-trained CNN model: Load the weights of the pre-trained CNN model that was trained on the FER 2013 dataset.<br><br>
Select sample images: Choose a few sample images from the dataset that represent the emotions of angry, happy, and sad.<br><br>
Extract feature maps: Pass the selected sample images through the pre-trained CNN model and extract the feature maps from intermediate layers. These feature maps capture the learned representations of different features in the images.<br><br>
Visualize the feature maps: Plot the feature maps for each emotion category (angry, happy, and sad). Each feature map represents a specific learned feature or pattern that the model uses to classify the emotions.<br><br>
Interpretation: Analyze the patterns and activations in the feature maps to gain insights into what the model focuses on when predicting different emotions. This can help in understanding which facial features or regions contribute to the model's decision-making process.


# Logistic-Regression-on-the-Titanic-Dataset
This project aims to predict the survival of passengers on the Titanic using logistic regression. The Titanic dataset is a well-known dataset that contains information about the passengers aboard the Titanic, including whether they survived or not.

### Dataset
The dataset used in this project is the Titanic dataset, which can be found in the file titanic.csv. It contains the following columns:

-PassengerId: Unique identifier for each passenger<br>
-Survived: Survival status (0 = No, 1 = Yes)<br>
-Pclass: Passenger class (1 = 1st class, 2 = 2nd class, 3 = 3rd class)<br>
-Name: Passenger's name<br>
-Sex: Passenger's sex (Male or Female)<br>
-Age: Passenger's age<br>
-SibSp: Number of siblings/spouses aboard<br>
-Parch: Number of parents/children aboard<br>
-Ticket: Ticket number<br>
-Fare: Fare paid for the ticket<br>
-Cabin: Cabin number<br>
-Embarked: Port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton)<br>

### Dependencies
The following libraries are used in this project:

-Pandas: For data manipulation and analysis<br>
-NumPy: For numerical operations<br>
-Scikit-learn: For machine learning algorithms<br>
-Matplotlib: For data visualization<br>
-Seaborn: For enhanced data visualization<br>

### Usage
Clone the repository:<br>
-git clone https://github.com/yourusername/Logistic-Regression-on-the-Titanic-Dataset.git

Install the required dependencies:<br>
-pip install pandas numpy scikit-learn matplotlib seaborn

Run the logistic_regression.py script:<br>
-python logistic_regression.py

### Results
The logistic regression model predicts the survival of passengers based on the provided features. The accuracy of the model on the test dataset is 72.97%. Additionally, various visualizations are generated to explore the dataset and gain insights into the relationships between different variables.


# Multiple Models Training on the Google Play Store Dataset
This project aims to train multiple models on the Google Play Store dataset to predict various app attributes and categories. The dataset contains information about different apps available on the Google Play Store, including app names, categories, ratings, reviews, sizes, and more.

## Dataset
The dataset used in this project is the Google Play Store dataset, which can be downloaded from Kaggle using the following link: <br>[Google Play Store Dataset](https://www.kaggle.com/datasets/gauthamp10/google-playstore-apps)<br>

### The dataset consists of the following columns:<br>

App: Name of the app<br>
Category: Category of the app<br>
Rating: Average user rating of the app<br>
Reviews: Number of user reviews for the app<br>
Size: Size of the app<br>
Installs: Number of app installations<br>
Type: Paid or Free<br>
Price: Price of the app (if it is a paid app)<br>
Content Rating: Content rating for the app<br>
Genres: Genres under which the app falls<br>
Last Updated: Date when the app was last updated<br>
Current Ver: Current version of the app<br>
Android Ver: Minimum Android version required to run the app<br>

## Data Preprocessing
Before training the models, it's essential to preprocess the dataset. This step involves handling missing values, converting categorical variables to numerical representations, scaling numerical features, and splitting the dataset into training and testing sets. You can refer to the notebook or script that accompanies this README for the specific data preprocessing steps.

## Choose Multiple Models
Select the machine learning or deep learning models you want to train on the Google Play Store dataset. Some popular models for regression or classification tasks include:

Logistic Regression
Random Forest
Gradient Boosting
Depending on your task (regression or classification) and preference, you can choose one or more models to train.

## Feature Engineering
If required, perform feature engineering on the dataset. This step involves selecting relevant features, creating new features, or transforming existing features to improve the model's performance. Feature engineering techniques include one-hot encoding, feature scaling, feature extraction, and more.

## Model Training
Train the selected models using the preprocessed dataset. Use appropriate model training techniques such as cross-validation, hyperparameter tuning, and regularization to optimize the model's performance. Evaluate the models using suitable evaluation metrics like mean squared error (MSE), accuracy, precision, recall, or F1-score.

## Model Comparison
Compare the performance of the trained models using the evaluation metrics.

