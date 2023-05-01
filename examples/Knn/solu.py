#!/usr/bin/python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotext as plo
from collections import Counter

df = pd.read_csv("data/DATA.csv",header=None)
li = {"Very Strong":[], "Strong":[],"Average":[],"Weak":[],}
new_feature = [1256,1]

for i in df.values:
  li[i[1]].append([i[0],i[2]])

[[plo.scatter([ii[0]],[ii[1]]) for ii in li[i]] for i in li]
plo.scatter([new_feature[0]],[new_feature[1]],color="g")
plo.show()
def knn(data, predict,k=3):
  distances = []
  for group in data:
    for features in data[group]:
      euclidean_distance = np.linalg.norm(np.array(features)-np.array(predict))
      distances.append([euclidean_distance,group])
  votes = [i[1] for i in sorted(distances)[:k]]
  print(votes)
  vote_result = Counter(votes).most_common(1)[0][0]
  return vote_result

result = knn(li, new_feature)
print(result)
