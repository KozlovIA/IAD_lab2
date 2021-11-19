




def readFile(file = "data.txt"):  # считывание из файла
    """Reading from file
    file - file name"""
    x = []
    file = open(file)
    for line in file:
        values = list(map(str, line.split()))
        temp = []
        for val in values:
            try:
                temp.append(float(val))
            except:
                temp.append(val)
        x.append(temp)
    return x

X = readFile("irisdat.txt")
for i in range(len(X)):
    print(X[i])


