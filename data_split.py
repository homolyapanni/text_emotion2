import math
import random

def split_data(dataset):
    
    train_ratio = 0.75
    test_ratio = 0.10
    
    n=len(dataset)
    random.shuffle(dataset)
    
    train_length=math.floor(n*0.75)
    val_length=math.floor(n*0.15)
    test_length=train_length+val_length
    
    train_set=dataset[:train_length]
    val_set=dataset[train_length:test_length]
    test_set=dataset[test_length:]
    
    return train_set, val_set, test_set
