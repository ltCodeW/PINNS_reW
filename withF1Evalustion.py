import numpy as np
import matplotlib.pyplot as plt
import re
def floatvalue(x):
    return float(x)
if __name__ == "__main__":
    pattern = "Loss: (\d[.]\d+e[+-]\d{2})"
    file = open("loss/trainProgress3.txt", "r")
    fileContent = file.read()

    loss = re.findall(pattern,fileContent,0)
    # intvalue1 = int(loss[1])
    # print(intvalue1)
    loss = list(map(floatvalue,loss))

    lossplt =loss[::1000]
    idx = range(len(lossplt))
    plt.savefig('./noSupervise.png')
    plt.plot(idx, lossplt)

    plt.show()
    print("debug")

