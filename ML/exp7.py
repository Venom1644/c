from sklearn.datasets import make_moons

X,y = make_moons(n_samples=3000, noise=0.2)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.4,
    random_state=50
)

from sklearn.neural_network import MLPClassifier

mlpc = MLPClassifier(
    hidden_layer_sizes=(5, 7),
    max_iter=2000,
    random_state=50,
    activation='logistic',
    learning_rate_init=0.0003
)

mlpc.fit(X_train, y_train)

y_pred = mlpc.predict(X_test)

from sklearn.metrics import accuracy_score

a = accuracy_score(y_pred, y_test)

print(a * 100)

