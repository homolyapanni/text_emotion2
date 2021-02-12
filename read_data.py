import csv

def read_data(path):
    data = []
    output=[]
    read = os.listdir(path)
    for fn in read:
        if fn.endswith(".csv"):
            with open(fn, 'r') as file:
                reader = csv.reader(file)
                for row in reader:
                    data.append(row)
                    
    for i in range(1,len(data)):
        output.append((data[i][3],data[i][1]))
        
    return output
