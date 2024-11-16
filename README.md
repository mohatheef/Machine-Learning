# Comparative Analysis of Machine Learning Classifiers on a Image Processing Dataset Using Wrapper Method Feature Selection:

1. Raw Data Visualization:
Purpose: To visualize the raw image data before any preprocessing.
Steps:
• Import necessary libraries (os, pandas, numpy, PIL, seaborn, matplotlib).
• Define paths to the image directory and the CSV file for saving the raw data.
• Traverse the image directory, load images, flatten them into pixel data, and store
them in a list.
• Create a pandas DataFrame from the image data list.
• Save the DataFrame to a CSV file.
• Visualize the class distribution, pixel intensity distribution of a sample image,
correlation of mean pixel intensities between classes, and a pairplot of image
statistics.

3. Pre-processing:
Purpose: To prepare the raw image data for model training.
Steps:
• Import necessary libraries (os, csv, PIL, torchvision).
• Define paths to the training directory and the output CSV file.
• Define a set of image transformations using torchvision.transforms (resizing,
converting to tensor, and normalizing).
• Create a function to preprocess images and save them into a CSV file.
• Run the preprocessing function and save the preprocessed data to the CSV file.

4. Pre-Processed Data Visualization:
Purpose: To visualize the preprocessed image data.
Steps:
• Import necessary libraries (pandas, numpy, matplotlib, seaborn).
• Load the preprocessed images CSV file into a pandas DataFrame.
• Visualize the class distribution, pixel intensity distribution of a sample image,
correlation of mean pixel intensities between classes, and a pairplot of image
statistics.

5. Feature Extraction:
Purpose: To extract relevant features from the images.
Steps:
• Import necessary libraries (os, numpy, pandas, cv2, tqdm, scipy, matplotlib).
• Define paths to the input and output directories.
• Create a function to extract features from an image (mean, standard deviation, etc.).
• Create a function to process a folder of images, extract features, and optionally
visualize them.
• Process all folders in the input directory and save the extracted features to CSV files.

6. Feature Extraction Visualization:
Purpose: To visualize the extracted features.
Steps:
• Import necessary libraries (seaborn, matplotlib, pandas).

• Create a function to visualize features from the extracted CSV file using histograms
with kernel density estimation (KDE).
• Call the visualization function to display the distributions of the extracted features.

7. Renaming target column:
Purpose: To rename the target column ('filename') in the extracted features CSV file to
"100."
Steps:
• Import pandas.
• Load the extracted features CSV file.
• Rename all rows in the 'filename' column to "100."
• Save the modified CSV file.

8. Feature Selection using Wrapper Method:
Purpose: To select the most relevant features for model training using a wrapper method
(Sequential Feature Selection with Random Forest).
Steps:
• Import libraries (pandas, numpy, matplotlib, seaborn, sklearn).
• Load the extracted features CSV file.
• Split the data into training and testing sets.
• Perform forward and backward feature selection using Random Forest as the
estimator.
• Evaluate the model performance with the selected features.
• Display the selected features and plot the accuracy of the different selection
methods.

9. Feature Extraction from Feature Selection:
Purpose: To extract the selected features from the original extracted features CSV files.
Steps:
• Import libraries (pandas, numpy, os, re).
• Define the directory containing the extracted features CSV files and the list of
features to extract.
• Loop through each CSV file, extract the specified features, and add a 'filename'
column.
• Concatenate all extracted features DataFrames into a single DataFrame.
• Save the final DataFrame to a new CSV file.

10. Classifier Training and Validation:
Purpose: This section focuses on training multiple machine learning classifiers using the
selected features and evaluating their performance on unseen data.
Steps:
• Import Libraries: Necessary libraries
like os, pickle, numpy, pandas, sklearn components (for classifiers, metrics,
preprocessing, and model selection), tqdm, and matplotlib are imported.
• Load Data: The extracted features are loaded from the CSV file
(extracted_features_final.csv).
• Data Splitting:

• The data is split into features (X) and target labels (y).
• Features are standardized using StandardScaler to ensure they have zero mean and
unit variance.
• The data is further divided into training and testing sets using train_test_split with an
80/20 ratio.

Classifier Initialization: Four classifiers are initialized:
• RandomForestClassifier
• SVC (Support Vector Classifier)
• KNeighborsClassifier
• MLPClassifier (Multi-layer Perceptron Classifier)
Training and Evaluation:
In a loop, each classifier is trained on the training data and then evaluated on the testing
data.
accuracy_score is used to calculate the accuracy of the model.
print_metrics function is called to compute and display detailed classification metrics
(confusion matrix, sensitivity, specificity, precision, F1-score, classification report).
The trained model is saved using pickle for later use.

ROC AUC (Receiver Operating Characteristic Area Under the Curve) and ROC curve are
plotted if the classifier supports probability estimates.

ROC Curve Plotting:
A ROC curve is plotted to visualize the performance of each classifier in terms of true
positive rate and false positive rate.

11. 10 flod Cross Validation
Classifier Initialization: Four classifiers are initialized:
• RandomForestClassifier
• SVC (Support Vector Classifier)
• KNeighborsClassifier
• MLPClassifier (Multi-layer Perceptron Classifier)
