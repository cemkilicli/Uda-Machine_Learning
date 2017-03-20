
import numpy as np
X = np.random.randint(6, size=(6, 10))  # assume that this is the survey outputs of 10 questions from 6 individuals
y = np.array([0, 1, 2, 3, 4, 1])  # assume that these are the political parties for 6 participants
print X
print X[2:3]
print y
from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB()
clf.fit(X, y)
MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True)
print(clf.predict(X[2:3]))
print(clf.predict_proba(X[2:3]))

A = np.array([1, 2, 3, 2, 3, 1, 3, 4, 3, 4])
pred2 = clf.predict(A)
print pred2
print clf.predict_proba(A)
