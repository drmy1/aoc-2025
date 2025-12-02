def findid():
    output = 0
    tmp = []
    with open("data.txt", encoding="utf-8") as f:
        for id in f.read().strip().split(","):
            startid = int(id.split("-")[0])
            endid = int(id.split("-")[1])
            for i in range(startid, endid + 1):
                if len(str(i)) > 1:
                    for k in range(2, len(str(i))+1):
                        if len(str(i)) % k == 0:
                            partlen = len(str(i)) // k
                            parts = []
                            for j in range(k):
                                parts.append(str(i)[j*partlen:(j+1)*partlen])
                            first = parts[0]
                            if all(p == first for p in parts):
                                print(f"ID: {i}")
                                print(f" Parts: {parts}")
                                print(f" Found matching ID: {i}")
                                output += i
                                break
                    
    return output
                    
if __name__ == "__main__":
    out = findid()
    print("Output:", out)