import math
data = [
    {"Income": 25000, "CreditScore": 600, "Student": "No", "Loan": "No"},
    {"Income": 30000, "CreditScore": 650, "Student": "No", "Loan": "No"},
    {"Income": 40000, "CreditScore": 700, "Student": "Yes", "Loan": "Yes"},
    {"Income": 50000, "CreditScore": 720, "Student": "Yes", "Loan": "Yes"},
    {"Income": 55000, "CreditScore": 690, "Student": "No", "Loan": "Yes"},
    {"Income": 60000, "CreditScore": 710, "Student": "No", "Loan": "Yes"},
    {"Income": 35000, "CreditScore": 640, "Student": "Yes", "Loan": "No"}
]

features = ["Income", "CreditScore", "Student"]
target = "Loan"

sample_1 = {"Income": 35000, "Creditscore": 690, "Student": "Yes"}

test_data_sample = [
    {"Income": 27000, "CreditScore": 620, "Student": "No"},
    {"Income": 42000, "CreditScore": 710, "Student": "Yes"},
    {"Income": 58000, "CreditScore": 700, "Student": "No"},
    {"Income": 36000, "CreditScore": 650, "Student": "Yes"}
]

class CART():
  def gini_index(self,rows, target):
    labels={}
    for row in rows:
      key=row[target]
      labels[key]=labels.get(key,0)+1
    
    gi=1
    total=len(rows)
    for value in labels.values():
      p=value/total
      gi-=p**2
    
    return gi
  
  def split_data(self,rows,feature,threshold):
    left=[]
    right=[]
    for row in rows:
      if row[feature]<=threshold:
        left.append(row)
      else:
        right.append(row)
    
    return left,right
  
  def gini_split(self,groups,target):
    total=sum(len(group) for group in groups)
    weighted_gi=0
    for group in groups:
      size=len(group)
      if size==0:
        continue
      weight=size/total
      gi_group=self.gini_index(group,target)
      weighted_gi+=weight*gi_group
    return weighted_gi
  
  def best_split(self,rows,features,target):
    best_feature=None
    best_threshold=None
    best_score=float("inf")
    best_groups=None
    for feature in features:
      values=set(row[feature] for row in rows)
      for threshold in values:
        left,right=self.split_data(rows,feature,threshold)
        groups=[left ,right]
        score=self.gini_split(groups,target)
        if score<best_score:
          best_score=score
          best_feature=feature
          best_threshold=threshold
          best_groups=groups
    
    return best_feature,best_threshold,best_groups
  
  def majority_class(self,rows,target):
    labels=[row[target] for row in rows]
    return max(set(labels),key=labels.count())
  
  def build_tree(self,rows,features,target):
    if len(rows)==0:
      return None
    
    labels=[row[target] for row in rows]
    if labels.count(labels[0])==len(labels):
      return labels[0]
    
    best_feature,best_threshold,best_groups=self.best_split(rows,features,target)
    if best_feature is None:
      return self.majority_class(rows,target)
    left,right=best_groups
    node={
      "feature":best_feature,
      "threshold":best_threshold,
      "left":self.build_tree(left,features,target),
      "right":self.build_tree(right,features,target)
    }
    return node
  
  def predict(self,node,row):
    if not isinstance(node,dict):
      return node
    
    feature=node["feature"]
    threshold=node["threshold"]
    if row[feature]<=threshold:
      return self.predict(node["left"],row)
    else:
      return self.predict(node["right"],row)
    
  def predict_dataset(self,tree,dataset):
    predictions=[]
    for row in dataset:
      predictions.append(self.predict(tree,row))
    return predictions


cart=CART()
gi=cart.gini_index(data,"Income")
feature,threshold,group=cart.best_split(data,features,target)
left,right=group
tree=cart.build_tree(data,features,target)
predict_1=cart.predict(tree,sample_1)
predict_test_data=cart.predict_dataset(tree,test_data_sample)
print("The gini index of the given feature:",gi)
print("best feature:",feature,"   best threshold:",threshold)
print(left)
print(right)
print("tree:",tree)
print("The prediction for the given sample:",predict_1)
print("The prediction for the given test dataset is:",predict_test_data)
    