import sys
from pathlib import Path
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
sys.path.insert(0,str(Path(__file__).resolve().parent))
from decision_tree_visualizer import build_tree, predict

data=load_iris(as_frame=True)
df=data.frame.rename(columns={'target':'Species'})
X_train,X_test,y_train,y_test=train_test_split(df.drop(columns='Species'),df['Species'],test_size=.2,random_state=42,stratify=df['Species'])
train=X_train.copy(); train['Species']=y_train
tree=build_tree(train,'Species',max_depth=5)
y_pred=predict(tree,X_test)
print('Dataset: Iris (150 rows, 4 numerical features, 3 classes)')
print('Training:',len(train),'Testing:',len(X_test),'random_state=42, stratified')
for label,score in [('Accuracy',accuracy_score(y_test,y_pred)),('Precision (macro)',precision_score(y_test,y_pred,average='macro',zero_division=0)),('Recall (macro)',recall_score(y_test,y_pred,average='macro',zero_division=0)),('F1 (macro)',f1_score(y_test,y_pred,average='macro',zero_division=0))]: print(label, f'{score:.4f}')
print(classification_report(y_test,y_pred,target_names=data.target_names,zero_division=0))
