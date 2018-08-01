
with open('test.txt','r') as f:
    for line in f.readlines():
        line = line[:-1]
        print(line)