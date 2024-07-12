# Import necessary libraries
from sklearn import tree
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

# Section 1: Basic Decision Tree Classification Example with Entropy Criterion
def basic_decision_tree_example():
    # Training data
    X = [[0, 0], [1, 1]]
    Y = [0, 1]

    # Initialize and train the classifier with entropy criterion
    clf = tree.DecisionTreeClassifier(criterion='entropy')
    clf = clf.fit(X, Y)

    # Predict the class of a new sample
    prediction = clf.predict([[2., 2.]])
    print("Prediction for [2., 2.]:", prediction)

    # Predict the probability of each class for a new sample
    probability = clf.predict_proba([[2., 2.]])
    print("Probability for [2., 2.]:", probability)

# Section 2: Decision Tree Classification with Iris Dataset using Entropy Criterion
def iris_decision_tree_example():
    # Load Iris dataset
    iris = load_iris()
    X, y = iris.data, iris.target

    # Initialize and train the classifier with entropy criterion
    clf = tree.DecisionTreeClassifier(criterion='entropy')
    clf = clf.fit(X, y)

    # Plot the trained decision tree
    plt.figure(figsize=(12, 8))
    tree.plot_tree(clf, filled=True)
    plt.show()

# Main execution
if __name__ == "__main__":
    print("Basic Decision Tree Example with Entropy Criterion:")
    basic_decision_tree_example()
    
    print("\nIris Decision Tree Example with Entropy Criterion:")
    iris_decision_tree_example()
