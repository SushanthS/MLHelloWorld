from sklearn import tree

#0 = apple
#1 = orange

#2 = smooth
#3 = bumpy


features = [[140,2],[130,2],[150,3],[170,3]]
labels = [0,0,1,1]

clf = tree.DecisionTreeClassifier()
clf = clf.fit(features, labels)
print(clf.predict([[130,3]]))
print(clf.predict([[150,2]]))
