
def file(start):
    cur = start
    with open("data.txt", encoding="utf-8") as f:
        zero = 0
        for line in f:
            line = line.strip()
            if not line:
                continue
            direction = line[0]
            steps = int(line[1:])

            # Count how many times the dial points at 0 during this rotation
            def hits_during_rotation(start, direction, steps):
                if steps == 0:
                    return 0

                if direction == "R":
                    k0 = (100 - (start % 100)) % 100
                else:  # L
                    k0 = start % 100
                if k0 == 0:
                    k0 = 100
                if k0 > steps:
                    return 0
                return 1 + (steps - k0) // 100

            zero += hits_during_rotation(cur, direction, steps)

            if direction == "L":
                cur = (cur - steps) % 100
            else:
                cur = (cur + steps) % 100
            
            
        
                
    print("Final value:", start)
    print("Zero hits:", zero)

if __name__ == "__main__":
    file(50)