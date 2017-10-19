# myTeam.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from captureAgents import CaptureAgent
import random, time, util
from game import Directions
import game
from util import nearestPoint
import random

#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first = 'OffensiveReflexAgent', second = 'OffensiveReflexAgent'):
  """
  This function should return a list of two agents that will form the
  team, initialized using firstIndex and secondIndex as their agent
  index numbers.  isRed is True if the red team is being created, and
  will be False if the blue team is being created.

  As a potentially helpful development aid, this function can take
  additional string-valued keyword arguments ("first" and "second" are
  such arguments in the case of this function), which will come from
  the --redOpts and --blueOpts command-line arguments to capture.py.
  For the nightly contest, however, your team will be created without
  any extra arguments, so you should make sure that the default
  behavior is what you want for the nightly contest.
  """

  # The following line is an example only; feel free to change it.
  return [eval(first)(firstIndex), eval(second)(secondIndex)]

##########
# Agents #
##########

class OffensiveReflexAgent(CaptureAgent):
  """
  A Dummy agent to serve as an example of the necessary agent structure.
  You should look at baselineTeam.py for more details about how to
  create an agent as this is the bare minimum.
  """

  def registerInitialState(self, gameState):
    """
    This method handles the initial setup of the
    agent to populate useful fields (such as what team
    we're on).

    A distanceCalculator instance caches the maze distances
    between each pair of positions, so your agents can use:
    self.distancer.getDistance(p1, p2)

    IMPORTANT: This method may run for at most 15 seconds.
    """

    '''
    Make sure you do not delete the following line. If you would like to
    use Manhattan distances instead of maze distances in order to save
    on initialization time, please take a look at
    CaptureAgent.registerInitialState in captureAgents.py.
    '''
    
    CaptureAgent.registerInitialState(self, gameState)

    '''
    Your initialization code goes here, if you need any.
    '''
    #self.start = gameState.getAgentPosition(self.index)
    
    if self.red:
        centralX = (gameState.data.layout.width - 2) / 2
    else:
        centralX = ((gameState.data.layout.width - 2) / 2) + 1
    self.boundary = []
    for i in range(1, gameState.data.layout.height - 1):
        if not gameState.hasWall(centralX, i):
            self.boundary.append((centralX, i))
    self.nearestFood = self.getFurthestTarget(gameState, gameState.getAgentState(self.index).getPosition(), self.getFood(gameState).asList())
    self.team = self.getTeam(gameState)
    self.opponent = self.getOpponents(gameState)
    self.randFoodStatus = 0
    self.randFood = random.choice(self.getFoodYouAreDefending(gameState).asList())
    if self.index == self.team[0]: 
        self.partnerIndex = self.team[1]
    else: 
        self.partnerIndex = self.team[0]

  def chooseAction(self, gameState):
    """
    Picks among actions randomly.
    """
    x, y = gameState.getAgentState(self.index).getPosition()
    myPos = (int(x), int(y))
    actions = gameState.getLegalActions(self.index)
    if len(actions) > 0:
        actions.remove(Directions.STOP)
    foods = self.getFood(gameState).asList()  
    capsules=self.getCapsules(gameState)
    foods += capsules
    partnerPos = gameState.getAgentState(self.partnerIndex).getPosition()

        
    if self.nearestFood not in foods: 
        mindis, self.nearestFood = self.getNearestTarget(gameState, myPos, foods)
            
    if len(foods) > 2: 
        
        scaredTimes = [gameState.getAgentState(i).scaredTimer for i in self.opponent]
        
        opponentGhosts = [i for i in self.opponent if not gameState.getAgentState(i).isPacman]
        opponentConfig = [gameState.getAgentState(i).configuration for i in opponentGhosts]
        
        if len(opponentGhosts) == 2: 
            #print("2")
            opponent1 = opponentConfig[0]
            opponent2 = opponentConfig[1]

            #if the pacman is chased
            beingChased=False
            if opponent1 is not None and opponent2 is not None and (scaredTimes[0] <= 5 or scaredTimes[1] <= 5): 
                
                opponent1Pos = opponent1.getPosition()
                opponent2Pos = opponent2.getPosition()
                
                if len(capsules) > 0:
                    distToCap, nearestCapsule = self.getNearestTarget(gameState, myPos, capsules)
                    if distToCap < self.getMazeDistance(opponent1Pos, nearestCapsule) & distToCap < self.getMazeDistance(opponent2Pos, nearestCapsule):
                        minDis, action = self.getBestAction(gameState, nearestCapsule, actions)
                        #print(self.index, "a", "cap", nearestCapsule, minDis)
                        return action
                ghostDis=min(self.getMazeDistance(myPos, opponent1Pos),self.getMazeDistance(myPos, opponent2Pos))

                #judge if pacman is being chased
                lastSaw = self.getPreviousObservation()
                myPos = gameState.getAgentState(self.index).getPosition()
                if lastSaw != None:
                    enemiesLast = [lastSaw.getAgentState(i) for i in self.getOpponents(gameState)]
                    inRangeLast = filter(lambda x: not x.isPacman and x.getPosition() != None and self.getMazeDistance(myPos,x.getPosition()) < 5 and x.scaredTimer < 5,enemiesLast)
                    if len(inRangeLast) > 0:
                        # being chased
                        lastDis=min(self.getMazeDistance(i.getPosition(),myPos) for i in inRangeLast)
                        if lastDis-ghostDis<=1 and ghostDis<3:
                            beingChased=True


                if ghostDis <= 5 and beingChased:
                    if gameState.getAgentState(self.index).isPacman:
                        #action = self.getBestActionAvoidTwoGhosts(gameState, opponent1Pos, opponent2Pos, actions)
                        minDis, nearestDoor = self.getNearestTarget(gameState, myPos, self.boundary)
                        minDis, action = self.getBestAction(gameState, nearestDoor, actions)
                        #print(self.index, "c", "door", nearestDoor, minDis)
                        return action
                    
                    
                    else: 
                        if self.isAtDoor(gameState) and self.randFoodStatus == 0: 
                            if len(self.getFoodYouAreDefending(gameState).asList()) > 0: 
                                self.randFood = random.choice(self.getFoodYouAreDefending(gameState).asList())
                            minDis, action = self.getBestAction(gameState, self.randFood, actions)
                            self.randFoodStatus = 6
                            #print(self.index, "b", "randfood", self.randFood, minDis)
                            return action

                        if self.getMazeDistance(myPos, partnerPos) <= 10: 
                            if self.index == self.team[0]:
                                action = self.getBestActionAvoidTwoGhosts(gameState, opponent1Pos, opponent2Pos, actions)
                                #minDis, nearestFood = self.getNearestTarget(gameState, myPos, foods)
                                #minDis, action = self.getBestAction(gameState, nearestFood, actions)
                                #print(self.index, "z", "suiside", nearestFood, minDis)
                                return action
                        if self.randFoodStatus == 0: 
                            if len(self.getFoodYouAreDefending(gameState).asList()) > 0: 
                                self.randFood = random.choice(self.getFoodYouAreDefending(gameState).asList())
                            minDis, action = self.getBestAction(gameState, self.randFood, actions)
                            self.randFoodStatus = 6
                            #print(self.index, "b-2", "randfood", self.randFood, minDis)
                            return action
                        else: 
                            minDis, action = self.getBestAction(gameState, self.randFood, actions)
                            self.randFoodStatus -= 1
                            #print(self.index, "countdown", self.randFood, minDis, self.randFoodStatus)
                            return action
                            
        elif len(opponentGhosts) == 1: 
            #print("1")
            opponent = opponentConfig[0]
            if opponent is not None and gameState.getAgentState(opponentGhosts[0]).scaredTimer <= 5:
                
                opponentPos = opponent.getPosition()
                
                if len(capsules) > 0:
                
                    distToCap, nearestCapsule = self.getNearestTarget(gameState, myPos, capsules)
                
                    if distToCap < self.getMazeDistance(opponentPos, nearestCapsule):
                        minDis, action = self.getBestAction(gameState, nearestCapsule, actions)
                        #print(self.index, "a", "cap", nearestCapsule, minDis)
                        return action
                
                if self.getMazeDistance(myPos, opponentPos) <= 5:
                
                    if gameState.getAgentState(self.index).isPacman: 
                        
                        #action = self.getBestActionAvoidOneGhost(gameState, opponentPos, actions)
                        minDis, nearestDoor = self.getNearestTarget(gameState, myPos, self.boundary)
                        minDis, action = self.getBestAction(gameState, nearestDoor, actions)
                        #print(self.index, "c-2", "door", nearestDoor, minDis)
                        return action
                    else: 
                        if self.getMazeDistance(myPos, partnerPos) <= 10: 
                            if self.index == self.team[0]: 
                                
                                action = self.getBestActionAvoidOneGhost(gameState, opponentPos, actions)
                                
                                #minDis, nearestFood = self.getNearestTarget(gameState, myPos, foods)
                                #minDis, action = self.getBestAction(gameState, nearestFood, actions)
                                #print(self.index, "z-2", "suiside", nearestFood, minDis)
                                return action
                        
                        if self.isAtDoor(gameState) and self.randFoodStatus == 0: 
                            if len(self.getFoodYouAreDefending(gameState).asList()) > 0: 
                                self.randFood = random.choice(self.getFoodYouAreDefending(gameState).asList())
                            minDis, action = self.getBestAction(gameState, self.randFood, actions)
                            self.randFoodStatus = 6
                            #print(self.index, "b-3", "randfood", self.randFood, minDis)
                            return action
                        
                        if self.randFoodStatus == 0:
                            if len(self.getFoodYouAreDefending(gameState).asList()) > 0: 
                                self.randFood = random.choice(self.getFoodYouAreDefending(gameState).asList())
                            minDis, action = self.getBestAction(gameState, self.randFood, actions)
                            self.randFoodStatus = 6
                            #print(self.index, "b-4", "randfood", self.randFood, minDis)
                            return action
                        else: 
                            minDis, action = self.getBestAction(gameState, self.randFood, actions)
                            self.randFoodStatus -= 1
                            #print(self.index, "countdown", self.randFood, minDis, self.randFoodStatus)
                            return action
        
        if self.randFoodStatus > 0: 
            minDis, action = self.getBestAction(gameState, self.randFood, actions)
            self.randFoodStatus -= 1
            #print(self.index, "countdown", self.randFood, minDis, self.randFoodStatus)
            return action
        
        partnerMinDisttoFood, partnerNearestFood = self.getNearestTarget(gameState, partnerPos, foods)
        myMinDisttoFood, myNearestFood = self.getNearestTarget(gameState, myPos, foods)
        minDisttoHome, nearestDoor = self.getNearestTarget(gameState, myPos, self.boundary)
        
        if gameState.getAgentState(self.index).numCarrying > minDisttoHome: 
            minDis, action = self.getBestAction(gameState, nearestDoor, actions)
            #print(self.index, "g", "door", nearestDoor, minDis) 
            return action
        
        if myNearestFood == partnerNearestFood:
            
            if myMinDisttoFood < partnerMinDisttoFood: 
                self.nearestFood = myNearestFood
            elif myMinDisttoFood == partnerMinDisttoFood:
                if self.index == self.team[0]: 
                    if self.getMazeDistance(myPos, self.nearestFood) <= myMinDisttoFood: 
                        self.nearestFood = self.getFurthestTarget(gameState, myPos, foods)
                else: 
                    self.nearestFood = myNearestFood
            else: 
                if self.nearestFood == myNearestFood: 
                    self.nearestFood = self.getFurthestTarget(gameState, myPos, foods)
        else:
            self.nearestFood = myNearestFood

        minDis, action = self.getBestAction(gameState, self.nearestFood, actions)
        #print(self.index, "h", "food", self.nearestFood, minDis) 
        return action
    
    else: 
        minDis, nearestDoor = self.getNearestTarget(gameState, myPos, self.boundary)
        minDis, action = self.getBestAction(gameState, nearestDoor, actions)
        #print(self.index, "i", "door", nearestDoor, minDis) 
        return action
            

  def getNearestTarget(self, gameState, pos, targets):
    minDis, nearestTarget = min([(self.getMazeDistance(pos, target), target) for target in targets])
    return (minDis, nearestTarget)
    
    
  def getFurthestTarget(self, gameState, pos, targets):
    maxDisttoTarget, furthestTarget = max([(self.getMazeDistance(pos, target), target) for target in targets])
    return furthestTarget

  def getSuccessor(self, gameState, action):
    """
    Finds the next successor which is a grid position (location tuple).
    """
    successor = gameState.generateSuccessor(self.index, action)
    pos = successor.getAgentState(self.index).getPosition()
    if pos != nearestPoint(pos):
        # Only half a grid position was covered
        return successor.generateSuccessor(self.index, action)
    else:
        return pos
  
  def getBestAction(self, gameState, targetPos, actions):
    minDis, bestAction = min([(self.getMazeDistance(self.getSuccessor(gameState, action), targetPos), action) for action in actions]) 
    #print("best action", minDis, action)
    #print([(self.getMazeDistance(self.getSuccessor(gameState, action), targetPos), action) for action in actions])
    return (minDis, bestAction)
  
  def getBestActionAvoidOneGhost(self, gameState, opponentPos, actions):
    maxDis, bestAction = max([(self.simulateAvoidOneGhost(gameState.generateSuccessor(self.index, action), 3, opponentPos), action) for action in actions]) 
    return bestAction
    
  def getBestActionAvoidTwoGhosts(self, gameState, opponent1Pos, opponent2Pos, actions):
    maxDis, bestAction = max([(self.simulateAvoidTwoGhosts(gameState.generateSuccessor(self.index, action), 3, opponent1Pos, opponent2Pos), action) for action in actions]) 
    return bestAction
  
  def isAtDoor(self, gameState):
    myPos = gameState.getAgentState(self.index).getPosition()
    if myPos in self.boundary: 
        return True
    else: 
        return False
        
  def simulateAvoidOneGhost(self, gameState, depth, opponentPos):
    #print("working1")
    state = gameState.deepCopy()
    
    if depth == 0:
        return self.getMazeDistance(state.getAgentPosition(self.index), opponentPos)
    else: 
        actions = state.getLegalActions(self.index)
        disToGhost = [self.simulateAvoidOneGhost(state.generateSuccessor(self.index, action), depth - 1, opponentPos) for action in actions]
        return sum(disToGhost)/len(disToGhost)
  
  def simulateAvoidTwoGhosts(self, gameState, depth, opponent1Pos, opponent2Pos):
    #print("working2")
    state = gameState.deepCopy()
    
    if depth == 0:
        return self.getMazeDistance(state.getAgentPosition(self.index), opponent1Pos) + self.getMazeDistance(state.getAgentPosition(self.index), opponent2Pos)
    else: 
        actions = state.getLegalActions(self.index)
        disToGhost = [self.simulateAvoidTwoGhosts(state.generateSuccessor(self.index, action), depth - 1, opponent1Pos, opponent2Pos) for action in actions]
        return sum(disToGhost)/len(disToGhost)
  
  
  
  
  
  
  