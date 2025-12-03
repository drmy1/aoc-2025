
def findbat():
    with open("data.txt") as f:
        outjoltage = 0
        for line in f.readlines():
            line = line.strip()
            a = [int(x) for x in line]
            if (len(a)-1) == a.index(max(a)):
                tmp = max(a)
                a.pop(a.index(max(a)))
                first = max(a)
                a.insert(len(a)+1, tmp)
            else:
                first = max(a)
            del a[0:a.index(first)+1]
            second = max(a)
            outjoltage += int(f"{first}{second}")
        print("Output:", outjoltage)

    
if __name__ == "__main__":
    findbat()