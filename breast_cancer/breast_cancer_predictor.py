# Imports machine learning classifier
from sklearn.neural_network import MLPClassifier

# Imports a dataset of breast cancer
from sklearn.datasets import load_breast_cancer

# Imports a function to train and split data sets
from sklearn.model_selection import train_test_split

# Load breast cancer dataset
breast_cancer_data = load_breast_cancer()

# Load a specific value of size of tumor
props = breast_cancer_data.data

# Load a value to tell if it's cancerous or not
isCancerous = breast_cancer_data.target

# Specific splitters for training and testing
attributes_train, attributes_test, labels_train, labels_test = train_test_split(props, isCancerous, test_size=0.4)


neuralnetwork = MLPClassifier(solver='lbfgs', activation='logistic', alpha=10.0)

# Function to train the neural network
neuralnetwork.fit(attributes_train, labels_train)

# Function to benchmark the accuracy scores
accuracy = neuralnetwork.score(attributes_test, labels_test) # Test the neural network

# Prints results
print(str(accuracy * 100) + "% of accuracy")