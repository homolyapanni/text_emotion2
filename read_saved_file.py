import os

def read_saved_file(path,filename):
    data=[]
    read = os.listdir(path)
    for fn in read:
        if fn == filename:
            with open(fn, 'r', encoding="utf-8") as file:
                reader = csv.reader(file)
                for row in reader:
                    if len(row) !=0:
                        data.append(tuple(row))
                    
    return data
