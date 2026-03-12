#for each feature, find entorpy , find gain inforamtion 
#compare the ig of features and select the feature which has lowest ig
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
    weighted_entorpy=0
    for subset in splits.values():
      p=len(subset)/len(rows)
      weighted_entorpy+=p*self.entropy(subset,target)
    ig=total_entropy-weighted_entorpy
    
    return ig
  



    
entro=DecisionTree()
result=entro.entropy(data,target)
splitted_data=entro.split_data(data,"Outlook")
info_gain=entro.info_gain(data,"Outlook",target)
print(result)
print(splitted_data)
print(len(splitted_data))
print(info_gain)
