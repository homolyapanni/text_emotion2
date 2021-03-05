import csv
import os

def read_data_N_rows(path,N):
    data = []
    output=[]
    read = os.listdir(path)
    for fn in read:
        if fn == "train.csv":
            with open(fn, 'r', encoding="utf-8") as file:
                reader = csv.reader(file)
                for i,row in enumerate(reader):
                    data.append(row)
                    if i == N:
                        break
    for i in range(1,len(data)):
        output.append((data[i][2],data[i][1]))
        
    with open('data_'+str(N)+'.csv','w',encoding='utf-8') as csv_file:  # elmenti az új adatot
        csv_writer = csv.writer(csv_file)
        for i in range(1,len(data)):
            csv_writer.writerow((data[i][2],data[i][1]))
            
    return output
