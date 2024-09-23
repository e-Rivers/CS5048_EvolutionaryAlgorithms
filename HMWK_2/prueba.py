from random import uniform
inputs = [5, 3, 2, 1, 6]
colors = ['Red', 'Blue', 'Orange', 'Green', 'Yellow']
def roulette_wheel(inputs):
    total_count = sum(inputs)
    probabilities = [round((float)(value)/total_count,2) for value in inputs]
    
    randomNumber = round(uniform(0, 1),3)
    
    cumulativeprobability = [probabilities[0]]
    for value in probabilities[1:]:
        cumulativeprobability.append(round(value + cumulativeprobability[-1],2))
        
    print("The probabilities of each color: ", probabilities)
    print("\nThe cumulative probabilities are: ", cumulativeprobability)
    choose = 0
    count = 0
    for value in probabilities:
        choose = choose + value
        count +=1
        if choose >= randomNumber:
            print("\nThe Random Number is: ", str(randomNumber) + '\n')
            print("Number Selected: " + str(value) + ". It's color is: " + colors[probabilities.index(value)])
            break

roulette_wheel(inputs)