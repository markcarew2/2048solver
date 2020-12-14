from BaseAI import BaseAI
import math
import numpy as np

#use time.process_time() this is cpu time its what the GameManager uses
import time

class PlayerAI(BaseAI):

    def getMove(self, grid):
        global startTime
        
        startTime = time.process_time()

        if grid.getAvailableMoves() == []:
            return None

        for depthLimit in range(1, 100):
            
            move = Minimax(grid, depthLimit)

            if move == None:
                break

            else:
                bestMove = move
                bestDepth = depthLimit

        print(bestDepth)
        return bestMove

class Node:
    def __init__(self, grid, depth, move = None):
        self.grid = grid
        self.depth = depth
        self.move = move



def Minimax(grid, depthLimit):
    global exploredMaxStates
    global exploredMinStates
    exploredMaxStates = {1:0}
    exploredMinStates = {1:0}
    
    fNode = Node(grid.clone(), 0)
    cuts = 0
    bestNode = MaxValue(fNode, -1000, 1000, depthLimit, cuts)

    if bestNode[1] == 0:
        return None
    
    return bestNode[0].move

def MaxValue(N, alph, beta, depthLimit, cuts):

    if((time.process_time()-startTime) > .19):
        return (None,0)

    if terminaltest(N.depth, depthLimit):
        return (None, eval(N.grid))
    
    #Get moves and orders based on the heuristic function
    moves = N.grid.getAvailableMoves()
    
    if moves == []:
        return (None, 0)

    orderedMoves = []
    for move in moves:
        test = N.grid.clone()
        test.move(move)
        value = eval(test)
        orderedMoves.append((move,value))
    
    orderedMoves.sort(key=takeSecond, reverse=True)
    orderedMoves = [ele[0] for ele in orderedMoves]

    (maxChild,maxUtility) = (None, -1000)

    for move in orderedMoves:
        child = Node(N.grid.clone(), N.depth + 1, move)
        child.grid.move(move)
        flattened = [val for sublist in child.grid.map for val in sublist]

        if tuple(flattened) in exploredMaxStates.keys():
            childUtility = exploredMaxStates[tuple(flattened)]
        else:
            _, childUtility = MinValue(child, alph, beta, depthLimit, cuts)

        if childUtility > maxUtility:
            (maxChild, maxUtility) = (child, childUtility)
        
        if maxUtility >= beta:
            cuts += 1
            #print(cuts)
            break
        
        if maxUtility > alph:
            alph = maxUtility

        key = tuple(flattened)
        exploredMaxStates[key] = childUtility

    return (maxChild, maxUtility)
    
def MinValue(N, alph, beta, depthLimit, cuts):

    #Check we have time
    if((time.process_time()-startTime) > .19):
        return (None,0)

    #Check if we've reached endgame
    if terminaltest(N.depth, depthLimit):
        return (None, eval(N.grid))

    #Get Moves
    moves = N.grid.getAvailableCells()

    if moves == []:
        return (None, 0)

    (minChild,minUtility) = (None, 1000)

    for move in moves:
        for value in [2,4]:
            child = Node(N.grid.clone(), N.depth + 1, move)
            child.grid.insertTile(move, value)
            flattened = [val for sublist in child.grid.map for val in sublist]

            if tuple(flattened) in exploredMinStates.keys():
                childUtility = exploredMinStates[tuple(flattened)]
            else:
                _, childUtility = MaxValue(child, alph, beta, depthLimit, cuts)

            if childUtility < minUtility:
                (minChild, minUtility) = (child, childUtility)
        
            if minUtility <= alph:
                cuts += 1
                #print(cuts)
                break
        
            if minUtility < beta:
                beta = minUtility

            flattened = [val for sublist in child.grid.map for val in sublist]
            key = tuple(flattened)
            exploredMinStates[key] = childUtility

    return (minChild, minUtility)

def terminaltest(depth, maxDepth):
    if depth >= maxDepth:
        return True
    else:
        return False

#heuristic function
def eval(grid):
    w1 = 1
    w2 = 1
    w3 = 2
    w4 = 1
    h = w1*emptyCells(grid) + w2*highestCell(grid)  + w3*ordering(grid) + w4*cellDistance(grid)
    return h

def emptyCells(grid):
    empty = grid.getAvailableCells()
    return len(empty)

def highestCell(grid):
    highest = 0
    for i in grid.map:
        for j in i:
            if j > highest:
                highest = j
    h = math.log2(highest)
    return h

def ordering(grid):
    ext = 2e-4
    map = np.array(grid.map)
    map = map + ext
    vectorizedLog2 = np.vectorize(math.log2)
    vectorizedLog2(map)
    maxID = np.argmax(map)
    if maxID == 0:
        multiplier = np.array([[40,15,5,3],[15,5,3,1],[5,3,1,0],[3,1,0,0]])
    elif maxID == 3:
        multiplier = np.array([[3,5,15,40],[1,3,5,15],[0,1,3,5],[0,0,1,3]])
    elif maxID == 12:
        multiplier = np.array([[3,1,0,0],[5,3,1,0],[15,5,3,1],[40,15,5,3]])
    elif maxID == 15:
        multiplier = np.array([[0,0,1,3],[0,1,3,5],[1,3,5,15],[3,5,15,40]])
    else:
        multiplier = np.array([[30,15,15,30],[15,5,5,15],[15,5,5,15],[30,15,15,30]])
    scoremap = np.multiply(map,multiplier)
    #print("Ordering value is: ", math.log2(np.sum(scoremap)))
    return math.log2(np.sum(scoremap))

def cellDistance(grid):
    map = np.array(grid.map)
    value = 0 
    for i, row in enumerate(map):
        for j, cell in enumerate(row):
            if j < 3 and i < 3 and cell > 0:
                if row[j+1] == cell or grid.map[i+1][j] == cell:
                    value += math.log2(cell)
    #print("Cell Distance Value is: ", value)
    return value


def flatten(listy):
    return [ele for sublist in listy for ele in sublist]

#Helper for ordering function
def takeSecond(elem):
    return elem[1]
