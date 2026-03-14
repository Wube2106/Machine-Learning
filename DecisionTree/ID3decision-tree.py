#for each feature, find entorpy , find gain inforamtion 
#compare the ig of features and select the feature which has highest ig
#then devide the dataset based on the selected feature
#recurisivel do thie task until agiven criteria is met.
import math
data = [
    {"Outlook": "Sunny", "Temperature": "Hot", "Humidity": "High", "Play": "No"},
    {"Outlook": "Sunny", "Temperature": "Hot", "Humidity": "Normal", "Play": "Yes"},
    {"Outlook": "Overcast", "Temperature": "Hot", "Humidity": "High", "Play": "Yes"},
    {"Outlook": "Rain", "Temperature": "Mild", "Humidity": "High", "Play": "Yes"},
    {"Outlook": "Rain", "Temperature": "Cool", "Humidity": "Normal", "Play": "Yes"},
    {"Outlook": "Rain", "Temperature": "Cool", "Humidity": "Normal", "Play": "No"},
    {"Outlook": "Overcast", "Temperature": "Cool", "Humidity": "Normal", "Play": "Yes"},
]
features = ["Outlook", "Temperature", "Humidity"]
target = "Play"
class DecisionTree():
  def __init__(self):
    pass

  def entropy(self,rows,target):
    label_count={}
    for row in rows:
      label=row[target]
      label_count[label]=label_count.get(label,0)+1
    ent=0
    total=len(rows)
    for count in label_count.values():
      p=count/total
      if p>0:
        ent-=p*math.log2(p)
    return ent
    
  def split_data(self,rows,feature):
    splits={}
    for row in rows:
      key=row[feature]
      if key  not in splits:
        splits[key]=[]
      splits[key].append(row)
    return splits
  
  def info_gain(self,rows,feature,target):
    splits=self.split_data(rows,feature)
    total_entropy=self.entropy(rows,target)
    weighted_entropy=0
    for subset in splits.values():
      p=len(subset)/len(rows)
      weighted_entropy+=p*self.entropy(subset,target)
    ig=total_entropy-weighted_entropy
    
    return ig
  
  def build_tree(self,rows,features,target):
    labels=[row[target] for row in rows]
    if labels.count(labels[0])==len(labels):
      return labels[0]
    if not features:
      return max(set(labels),key=labels.count)
    
    best=max(features,key=lambda f: self.info_gain(rows,f,target))
    tree={best:{}}
    remaining_features=[f for f in features if f !=best]
    
    for value,subset in self.split_data(rows,best).items():
      tree[best][value]=self.build_tree(subset,remaining_features,target)
    return tree

  def predict(self,tree,sample,default=None):
    if not isinstance(tree,dict):
      return tree
    
    feature=next(iter(tree))
    value_sample=sample.get(feature)

    sub_tree=tree[feature].get(value_sample)
    if sub_tree is None:
      return default
    
    return self.predict(sub_tree,sample,default)
    
entro=DecisionTree()
entropy_result=entro.entropy(data,target)
splitted_data=entro.split_data(data,"Outlook")
information_gain=entro.info_gain(data,"Outlook",target)

tree=entro.build_tree(data,features,target)
labels=[row[target] for row in data]
majority_class=max(set(labels),key=labels.count)

sample = {
    "Outlook": "Sunny",
    "Temperature": "Hot",
    "Humidity": "High"
}


prediction=entro.predict(tree,sample,majority_class)
print(entropy_result)
print(splitted_data)
print(information_gain)
print(tree)
print(prediction)



