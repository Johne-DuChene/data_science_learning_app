from flask import Flask

# initialize the app
app = Flask(__name__)

# execute iris function at /iris route
@app.route("/iris")
def iris():
    from sklearn.datasets import load_iris
    from sklearn.linear_model import LogisticRegression
    X, y = load_iris(return_X_y=True)
    clf = LogisticRegression(
        random_state = 42,
        solver="lbfgs",
        multi_class="multinomial"
    ).fit(X, y)

    return str(clf.predict(X[:2, :]))