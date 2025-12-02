
def file(start):
    cur = start
    with open("data.txt", encoding="utf-8") as f:
        zero = 0
        for line in f:
            line = line.strip()
            if line.startswith("L"):
                number = line[1:]
                cur -= int(number)
            if line.startswith("R"):
                number = line[1:]
                cur += int(number)
            
            cur = cur % 100
            if cur == 0:
                zero += 1
            
            
        
                
    print("Final value:", start)
    print("Zero hits:", zero)

if __name__ == "__main__":
    file(50)