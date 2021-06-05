f = open('original.txt', 'r')
ff = open('original2.txt','w')

bb=0 
lines = f.readlines()
for line in lines:
    ff.write(line)
    bb+=1
    if(bb==1000):
        break

f.close()
ff.close()
