import numpy as np
import matplotlib.pyplot as plt
def m2(x):
    return x*x
if __name__ == "__main__":
    list1=range(30)
    list2=list(map(m2,list1))
    plt.plot(list1, list2)

    plt.show()
    print("debug")



