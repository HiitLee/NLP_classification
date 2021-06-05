import csv

f = open('ratings_test.txt', 'r', encoding='utf-8', newline='')
lines = csv.reader(f, delimiter='\t')


ff = open('ratings_test.tsv', 'w', encoding='utf-8', newline='')
wr = csv.writer(ff, delimiter='\t')


length=0
data_good = 0
data_bad = 0
for line in lines:
    length+=1
    if(line[2] == '1'):
        data_good+=1
    elif(line[2] == '0'):
        data_bad+=1
    wr.writerow([line[2],line[1]])

f.close()
ff.close()

print('length:', length)
print('data_good:', data_good)
print('data_bad:', data_bad)
#wr = csv.writer(f, delimiter='\t')
