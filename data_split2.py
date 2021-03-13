import math
import random

def split_data(dataset):
    
    n=len(dataset)
    random.shuffle(dataset)
    
    train_length=math.floor(n*0.75)
   
    train_set=dataset[:train_length]
    val_set=dataset[train_length:]
    
    
    return train_set, val_set
