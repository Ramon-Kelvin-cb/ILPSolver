from scipy.optimize import linprog
import numpy as np

#Program to solve problems of the type: Max Objective subject to Weights <= Limits and all variables are positive integers

#Function that verifies if a number is an Integer
def isInteger(value) -> bool:
    if (value % 1 == 0):
        return True
    else:
        return False

#This function will return an array of feasible integer solutions to our ILP problem
def discretizeSolutions(objective, weights, limits, binary):
    #Non negativity of the variables (For 0-1 problems we just need to set the boundaries to [0,1])
    if binary:
        bnd = [(0, 1) for i in range(len(objective))]
    else:
        bnd = [(0, np.inf) for i in range(len(objective))]

    #The linprog method minimizes the objective function, so we need to adjust the values of the coeficients
    updatedObjective = [-value for value in objective]

    #Doing the simplex method to find the Real variables that optimizes our problem
    optimize = linprog(updatedObjective, A_ub= weights, b_ub= limits, bounds= bnd)

    #The problem has no feasible solution
    if not optimize.success:
        return []

    #The problem has a feasible solution
    else:
        #Getting the references to the real variables that solve our ILP problem
        realVariablesIndexes = [i for i,value in enumerate(optimize.x) if not isInteger(value)]

        #If there are no real varibles in our solution, the ILP problem is solved
        if not realVariablesIndexes:
            return [optimize.x]

        #There is at least one real variable in our solution
        else:
            #Choosing one real variable to branch on
            variableToBranch = realVariablesIndexes[0]

            #Creating the inequality (0.x1 + 0.x2 + ... + 1.xk + 0.x(k+1) + ... + 0.xn)
            newRestriction = np.zeros(len(objective))
            newRestriction[variableToBranch] = 1

            #Creating bounds
            boundBelow = np.floor(optimize.x[variableToBranch])
            boundAbove = np.ceil(optimize.x[variableToBranch])

            #Adding the restrictions to our problems
            problemBelowWeights = weights.copy()
            problemBelowWeights.append(newRestriction)
            problemBelowLimits = limits.copy()
            problemBelowLimits.append(boundBelow)

            problemAboveWeights = weights.copy()
            problemAboveWeights.append([-value for value in newRestriction])
            problemAboveLimits = limits.copy()
            problemAboveLimits.append(-boundAbove)

            #Recursive Call
            return discretizeSolutions(objective= objective, weights= problemBelowWeights, limits= problemBelowLimits, binary= binary) + discretizeSolutions(objective= objective, weights= problemAboveWeights, limits= problemAboveLimits, binary= binary)


#Returns the optimal solution to our ILP problem and the optimal value associated
def ilpSolver(objective, weights, limits, binary):
   solution = discretizeSolutions(objective= objective, weights = weights, limits = limits, binary = binary)

   bestSolution = max(solution, key = lambda el: np.dot(el, objective))
   optimalValue = np.dot(objective, bestSolution)

   return bestSolution, optimalValue



objective = [170000, 125000, 200000, 150000, 90000, 70000]
weights = [[600000, 250000, 750000, 200000, 250000, 100000]]
limits = [1250000]

bestSolution, optimalValue = ilpSolver(objective= objective, weights= weights, limits= limits, binary= False)
print("The best solution was found in: {b1}\nAnd the Optimal Value is: {b2}".format(b1 = bestSolution, b2 = optimalValue))
