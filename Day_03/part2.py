def findbat():
    with open("data.txt") as f:
        outjoltage = 0
        for line in f.readlines():
            line = line.strip()            
            digits = [int(x) for x in line]
            tmpjoltagebank = ""
            sidx = 0  # Start searching index
            needed = 12 # Number of digits I need
            for i in range(needed):
                remaining = needed - 1 - i # How many batteries remain to find
                endinx = len(digits) - remaining  # floating end index pointing at the end of valid search zone
                searchz = digits[sidx : endinx]  # valid search zone
            
                maxdig = max(searchz) # max dgit from the search zone
                maxidx = searchz.index(maxdig) # index of the found max digit
                
                tmpjoltagebank += str(maxdig) #! I know I coudnt make up better variable name :)
                
                sidx = sidx + maxidx + 1 # Update of start index for next search
            
            outjoltage += int(tmpjoltagebank)
            
        print("Output:", outjoltage)

    
if __name__ == "__main__":
    findbat()