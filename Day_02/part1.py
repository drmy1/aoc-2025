def findid():
    output = 0
    with open("data.txt", encoding="utf-8") as f:
        for id in f.read().strip().split(","):
            startid = int(id.split("-")[0])
            endid = int(id.split("-")[1])
            for i in range(startid, endid + 1):
                if len(str(i)) > 1:
                    firsthalf = str(i)[: len(str(i)) // 2]
                    secondhalf = str(i)[len(str(i)) // 2 :]
                    if firsthalf == secondhalf:
                        print(f"ID: {i}")
                        print(f" First half: {firsthalf}, Second half: {secondhalf}")
                        print(f" Found matching ID: {i}")
                        output += i
    return output
                    
if __name__ == "__main__":
    out = findid()
    print("Output:", out)