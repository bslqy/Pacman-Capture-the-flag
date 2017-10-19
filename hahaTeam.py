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

#TODO LIST:
"""
-add inference to state space search
    save return val of positionMoveInfer in State, use that as distrib. Might need to change mDistribs a lot, idk
-weight going home higher if near enemy ghost
-maybe weight running from ghosts higher
-Add short circuit in state space search, where if can eat food safely, eat the food always. there's not a better move there
-prune action sequences that takes us into many repeated positions when there isnt a significant increase in utility
"""

from captureAgents import CaptureAgent
import random, time, util
from game import Directions
import game
from capture import GameState, SIGHT_RANGE
from collections import namedtuple
from time import time

#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first='RealAgent', second='RealAgent'):
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

class DummyAgent(CaptureAgent):
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

    def chooseAction(self, gameState):
        """
        Picks among actions randomly.
        """
        actions = gameState.getLegalActions(self.index)

        '''
        You should change this in your own agent.
        '''

        return random.choice(actions)





class TeamData:
    RedData=None
    BlueData=None
    def __init__(self, gameState, team, opps, agent):
        self.team=team
        self.mAgent=agent
        self.offensive = True
        self.mDistribs=[None]*gameState.getNumAgents()
        #opps=self.team[0].getOpponents(gameState)
        for i in range(gameState.getNumAgents()):
            if i in opps:
                dist=util.Counter()
                oppPos=gameState.getInitialAgentPosition(i)
                dist[oppPos]=1.0
                self.mDistribs[i]=dist
            else:
                self.mDistribs[i]=None
        #should be all legal positions
        self.legalPositions = gameState.data.layout.walls.asList(key = False)
        self.defendFoodGrid=[]
        halfway = agent.getFood(gameState).width / 2
        self.borderPositions=[(halfway, y) for y in range(agent.getFood(gameState).height) if not gameState.hasWall(halfway, y)]

        self.borderDistances={}
        self.calcBorderDistances(gameState)
        #self.consideredStates = {}

    def calcBorderDistances(self, gameState):
        grid = gameState.getWalls()
        halfway = grid.width / 2
        if not self.mAgent.red:
            xrange = range(halfway)
        else:
            xrange = range(halfway, grid.width)

        for x in xrange:
            for y in range(grid.height):
                if not grid[x][y]:
                    self.borderDistances[(x, y)]= min(self.mAgent.getMazeDistance((x, y), borderPos) for borderPos in self.borderPositions)

    def logFood(self, gameState):
        self.defendFoodGrid.append(self.mAgent.getFoodYouAreDefending(gameState))

    def getOffensive(self):
        self.offensive = not self.offensive
        return self.offensive


class RealAgent(CaptureAgent):
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
        #set up data repository
        if self.red:
            if not TeamData.RedData:
                TeamData.RedData=TeamData(gameState, self.getTeam(gameState), self.getOpponents(gameState), self)
            self.data=TeamData.RedData

        else:
            if not TeamData.BlueData:
                TeamData.BlueData=TeamData(gameState, self.getTeam(gameState), self.getOpponents(gameState), self)
            self.data=TeamData.BlueData

        self.legalPositions=self.data.legalPositions
        self.offensive = self.data.getOffensive()
        #set up distribution list that will hold belief distributions for agents

    def chooseAction(self, gameState):

        '''
        You should change this in your own agent.
        '''
        #startTime=time()
        self.knownEnemies={}
        self.data.logFood(gameState)
        self.updatePosDist(gameState)
        if self.getScaredMovesRemaining(gameState)>2:
            self.offensive=True
        elif self.getScaredMovesRemaining(gameState)==1:
            self.offensive=self.data.getOffensive()
        ##print "infer time: ", time()-startTime
        self.displayDistributionsOverPositions(self.data.mDistribs)
        bestAction, utility= self.actionSearch(self.index, gameState)
        newPos=game.Actions.getSuccessor(self.getMyPos(gameState), bestAction)
        currFeatures=self.getFeatures(gameState)
        nextGameState=gameState.generateSuccessor(self.index, bestAction)
        nextFeatures=self.getFeatures(nextGameState)
        currWeight=self.getWeights(currFeatures, gameState)
        nextWeight=self.getWeights(nextFeatures, nextGameState)

        #print "offensive?", self.offensive
        if not self.offensive:

            currUtility=self.Utility(gameState, currFeatures, True)

            nextUtility=self.Utility(nextGameState, nextFeatures, True)

            enemyPac=[i for i in self.getOpponents(gameState) if gameState.getAgentState(i).isPacman]
            if len(enemyPac)>0:
                enemyPac=enemyPac[0]

        return bestAction

    def actionSearch(self, agentIndex, gameState):
        ##do a breadth first search until time runs out
        #dictionary to keep track of visited spots so we can look up their utility in constant time
        visited = dict()
        #set to keep track of
        visitedInSequence = [self.getMyPos(gameState)]
        #queue of action states to visit
        toVisit = util.Queue()
        actions = []
        #lower bound and upper bound set to arbitrary values for testing purposes

        start_time = time()
        debug = False


        currGameStateFeatures = self.getFeatures(gameState)
        currGameStateWeights = self.getWeights(currGameStateFeatures, gameState)
        closestHomeAction=None
        closestHomeDistance=5000
        #do some quick thinking on this state first, check for obvious good moves
        for action in gameState.getLegalActions(self.index):
            newPos=game.Actions.getSuccessor(self.getMyPos(gameState), action)
            try:
                if self.getFood(gameState)[int(newPos[0])][int(newPos[1])] and (min([self.getDistanceToEnemy(gameState, i) for i in self.getOpponents(gameState)])>3 or self.getEnemyAgentScaredMovesRemaining(gameState)>3):
                    #print "foodShort"
                    return action, 0
            except:
                pass
            if self.onMySide(gameState, newPos) and self.getFoodEatenBySelf(gameState)>0:
                #print "home short"
                return action, 0
            if self.getScaredMovesRemaining(gameState)==0 and newPos in self.knownEnemies.values() and self.onMySide(gameState, newPos):
                #print "eat enemy short"
                return action, 0


        bestActionSequence = gameState.getLegalActions(self.index)
        bestActionSequenceUtility = None


        maxSearchDepth = 0
        consideredStates = []

        #named tuple for readability
        State = namedtuple('State', 'agentIndex actions visitedInSequence currGameState currGameStateFeatures totalUtility mDistribs')
        toVisit.push((State(agentIndex, actions, visitedInSequence, gameState, currGameStateFeatures, 0, self.data.mDistribs), 0))
        #using a constant of .75 seconds for now
        total_search_time = 0
        numberofsearches = 0
        while time() - start_time < .85 and not toVisit.isEmpty():
            curr_state, curr_utility = toVisit.pop()
            repeated_state = False
            legal_actions = curr_state.currGameState.getLegalActions(self.index)
            if len(legal_actions) == 1:
                repeated_state = True
            for next_action in legal_actions:
                st = time()
                if next_action=="Stop":
                    continue
                next_game_state = curr_state.currGameState.generateSuccessor(curr_state.agentIndex, next_action)
                my_pos = self.getMyPos(next_game_state)

                if repeated_state or my_pos not in curr_state.visitedInSequence:
                    new_visited = list(curr_state.visitedInSequence)
                    new_visited.append(my_pos)

                    new_actions = [action for action in curr_state.actions]
                    new_actions.append(next_action)
                    consideredStates.append(new_actions)

                    curr_state_features = curr_state.currGameStateFeatures

                    next_state_features = self.getFeatures(next_game_state)

                    #TODO: test the code below
                    if next_state_features["distToEnemyGhost"]<=len(new_actions) and next_game_state.getAgentState(self.index).isPacman and len(new_actions)<6 and self.getEnemyAgentScaredMovesRemaining(gameState)==0:
                        #continue, we're too close to an enemy ghost
                        #print "too close to ghost circuit"
                        continue



                    if next_game_state in visited:
                        state_utility = visited[next_game_state]
                    else:
                        state_utility = self.Utility(next_game_state, next_state_features)
                        visited[next_game_state] = state_utility

                    total_utility = state_utility + curr_state.totalUtility

                    toVisit.push((State(agentIndex, new_actions, new_visited, next_game_state, next_state_features, total_utility, self.data.mDistribs), total_utility/len(new_actions)))
                    total_search_time += time() - st
                    numberofsearches += 1

        while not toVisit.isEmpty():
            state, avg_utility = toVisit.pop()
            if not bestActionSequenceUtility or avg_utility > bestActionSequenceUtility:
                bestActionSequenceUtility = avg_utility
                bestActionSequence = state.actions

        return bestActionSequence[0], bestActionSequenceUtility

    #Doing a recalculation of minimum distance to food here - we can get rid of this later
    def estimatedUtilityWillIncrease(self, curr_state_features, next_state_features):
        if self.offensive:
            return next_state_features["foodDist"] < curr_state_features["foodDist"] or next_state_features["foodEatenBySelf"] > curr_state_features["foodEatenBySelf"]
        if next_state_features["numEnemyPacmen"] == 0:
            return True
        if curr_state_features["numEnemyPacmen"] == 0:
            return False
        for i in range(min(len(curr_state_features["distToEnemyPacman"]),len(next_state_features["distToEnemyPacman"]))):
            if next_state_features["distToEnemyPacman"][i] < curr_state_features["distToEnemyPacman"][i]:
                return True
        return False


    def Utility(self, gameState, features, debug=False):

        #features = self.getFeatures(gameState)
        weights = self.getWeights(features, gameState)

        utility = 0
        for feature, feature_value in features.items():

            if not feature_value or not weights[feature]:
                continue

            if isinstance(feature_value,list):
                if feature=="distToEnemyPacman":
                    for i in range(len(feature_value)):
                        if not feature_value[i]:
                            #we ate a pacman, add utility for this
                            utility+=6*weights[feature]
                            continue
                        utility+=6./feature_value[i] * weights[feature]
                else:
                    for i in range(len(feature_value)):
                        utility += feature_value[i] * weights[feature]
            elif feature=="foodDist":
                utility+=5./feature_value*weights[feature]
            elif feature=="distToNearestCapsule":
                utility+=4./feature_value*weights[feature]
            elif feature=="distToEnemyGhost":
                utility+=6./feature_value * weights[feature]
            elif feature=="distToHome":
                utility+=4./feature_value*weights[feature]
            elif feature=="distToNearestTeammate":
                utility+=2./feature_value*weights[feature]
            else:
                utility += feature_value * weights[feature]
        return utility

    #weight on a -5 to 5 scale
    def getWeights(self, features, gameState):
        def getEnemyGhostDistanceDistrib(distance):

            if 0<=distance <=2:
                return -7
            elif 2<distance<=4:
                return -1
            elif 4<distance<=6:
                return -.2
            else:
                return 0

        def getDistToHomeDistrib(weights2):
            if weights2["foodEatenBySelf"]==0:
                return 0
            if features["scaredEnemyMovesRemaining"]>(features["distToHome"]+3):
                return 0
            elif self.getScore(gameState)<0:
                #we losing
                if features["foodEatenBySelf"]>2:

                    return -4
                elif features["foodEatenBySelf"]>0:
                    return -2
            elif self.getScore(gameState)>4:
                return -.75*features["foodEatenBySelf"]
            elif features["foodEatenBySelf"]>3:
                return -1.5*features["foodEatenBySelf"]
            else:
                return -1*features["foodEatenBySelf"]

        weights = util.Counter()
        weights["foodDist"] = 3 if self.offensive else 1
        weights["distToNearestCapsule"]= 1 if features["distToEnemyGhost"]>3 else 2  #if closer to enemy, better eat the capsule
        weights["numEnemyPacmen"] = 0
        weights["distToEnemyPacman"] = 0 if features["numEnemyPacmen"]==0 else 2 if self.offensive else 4 if features["distToEnemyPacman"] > features["scaredMovesRemaining"] else -3
        if features["scaredMovesRemaining"] ==0 and weights["distToEnemyPacman"]>0:
            if not self.offensive:
                weights["foodDist"]=0
                weights["distToNearestCapsule"]=0
            if 5>features["enemyPacmanFood"]>=3:
                weights["distToEnemyPacman"]+=2
            elif features["distToEnemyPacman"]>=5:
                weights["distToEnemyPacman"]+=3
                weights["foodDist"]-=1
        weights["numEnemyGhost"] = 0
        weights["distToEnemyGhost"] = 0 if not gameState.getAgentState(self.index).isPacman else 1 \
                    if 0< features["scaredEnemyMovesRemaining"] <= features["distToEnemyGhost"] else getEnemyGhostDistanceDistrib(features["distToEnemyGhost"])
        weights["score"] = .5
        weights["movesRemaining"] = 0
        weights["scaredMovesRemaining"] = 0
        weights["scaredEnemyMovesRemaining"]=.5
        weights["foodEatenBySelf"] = 5 if self.offensive else 0 if weights["distToEnemyPacman"]>0 else 4
        weights["enemyPacmanFood"] = 0

        weights["distToNearestTeammate"] = -2
        weights["distToHome"] = getDistToHomeDistrib(weights)
        if not self.offensive and features["numEnemyPacmen"]>0:
            weights["distToHome"]+=1
        if weights["distToHome"]<-3:
            weights["foodDist"]=0
            weights["foodEatenBySelf"]=0
            weights["distToNearestCapsule"]=0
        elif weights["distToHome"]<=-2:
            weights["foodDist"]-=1
            weights["foodEatenBySelf"]-=1

        return weights



    def getFeatures(self, gameState):
        features=util.Counter()
        myPos= self.getMyPos(gameState)
        enemies=[gameState.getAgentState(i) for i in self.getOpponents(gameState)]


        foodList=self.getFood(gameState).asList()

        if len(foodList):
            minDist=self.getMazeDistance(myPos, foodList[0])
            for pos in foodList:
                dist=self.getMazeDistance(myPos, pos)
                if dist<minDist:
                    minDist=dist
                    if dist==1:
                        break
            features["foodDist"]=minDist
        else:
            features["foodDist"]=0
        features["numEnemyPacmen"]=0
        features["distToEnemyPacman"]=[]
        features["numEnemyGhost"]=0
        features["distToEnemyGhost"]=50000
        for i, enemy in zip(self.getOpponents(gameState), enemies):
            if enemy.isPacman:
                features["numEnemyPacmen"]+=1
                features["distToEnemyPacman"].append(self.getDistanceToEnemy(gameState, i))

            else:
                features["numEnemyGhost"]+=1
                features["distToEnemyGhost"]=min(self.getDistanceToEnemy(gameState, i), features["distToEnemyGhost"])

        features["score"]=self.getScore(gameState)
        features["movesRemaining"]=gameState.data.timeleft

        features["distToNearestCapsule"]=self.getDistToNearestCapsule(gameState)
        features["scaredMovesRemaining"]=self.getScaredMovesRemaining(gameState)
        features["scaredEnemyMovesRemaining"] = self.getEnemyAgentScaredMovesRemaining(gameState)
        features["foodEatenBySelf"]=self.getFoodEatenBySelf(gameState)
        features["enemyPacmanFood"]=[]
        for i in self.getOpponents(gameState):
            features["enemyPacmanFood"]=self.getFoodEatenByEnemyAgent(gameState, i)
        features["distToHome"]=self.getDistanceToHomeSide(gameState)
        features["distToNearestTeammate"] = self.getDistToNearestTeammate(gameState)

        return features

    def getDistToNearestTeammate(self, gameState):
        return min([self.getMazeDistance(gameState.getAgentState(self.index).getPosition(), gameState.getAgentState(agent).getPosition()) for agent in self.getTeam(gameState) if agent != self.index])

    def getDistToNearestCapsule(self, gameState):
        try:
            return min([self.getMazeDistance(gameState.getAgentState(self.index).getPosition(), cap) for cap in self.getCapsules()])
        except ValueError:
            return 0

    def getScaredMovesRemaining(self, gameState):
        return gameState.getAgentState(self.index).scaredTimer

    def getEnemyAgentScaredMovesRemaining(self, gameState):
        return gameState.getAgentState(self.getOpponents(gameState)[0]).scaredTimer

    def getFoodEatenByEnemyAgent(self, gameState, agentIndex):
        return gameState.getAgentState(agentIndex).numCarrying

    #food in our stomach
    def getFoodEatenBySelf(self, gameState):
        return gameState.getAgentState(self.index).numCarrying

    #gain of going to home side

    def getDistanceToHomeSide(self, gameState):
        myPos=self.getMyPos(gameState)
        return self.data.borderDistances[myPos] if myPos in self.data.borderDistances else 0



    def getMyPos(self, gameState):
        return gameState.getAgentPosition(self.index)

    def getDistanceToEnemy(self, gameState, enemyIndex):
        if enemyIndex in self.knownEnemies:
            return self.getMazeDistance(self.getMyPos(gameState), self.knownEnemies[enemyIndex])
        else:
            #list of all distances that have prob>=.5
            dists=[self.getMazeDistance(self.getMyPos(gameState), pos)
                   for pos, prob in self.getmDistribs(enemyIndex).items() if prob>=.5]
            if len(dists)==0:
                try:
                    bestPos= max(self.getmDistribs(enemyIndex).items(),
                           key= lambda x : x[1])[0]
                    while bestPos not in self.legalPositions:
                        self.getmDistribs(enemyIndex).pop(bestPos)
                        #print "bestPos illegal?", bestPos
                        bestPos= max(self.getmDistribs(enemyIndex).items(),
                           key= lambda x : x[1])[0]
                except ValueError:
                    return 6
                try:
                    return self.getMazeDistance(self.getMyPos(gameState), bestPos)
                except Exception:

                    return 2
            return sum(dists)/len(dists)


    def _setKnownPosDist(self, agentIndex, knownPos):
        self.knownEnemies[agentIndex]=knownPos
        dist=self.getmDistribs(agentIndex)
        dist.clear()
        dist[knownPos]=1.0

    def _capturedAgent(self, agentIndex):
        self._setKnownPosDist(agentIndex, self.getCurrentObservation().getInitialPosition(agentIndex))

    def getPrevPlayer(self):
        return (self.index-1)%self.getCurrentObservation().getNumAgents()

    #if this causes a significant slowdown, can just add mDistribs attribute to this class, initialize as ref to data
    def getmDistribs(self, agentIndex):
        return self.data.mDistribs[agentIndex]

    def setDistrib(self, agentIndex, newDistrib):
        self.data.mDistribs[agentIndex]=newDistrib

    #checks which side of the board a given position is, returns true if its my side
    def onMySide(self, gameState, pos):
        halfway = gameState.data.food.width / 2
        #copied from halfgrid
        if self.red:
            return pos[0] < halfway
        else:
            return pos[0] > halfway


    #does inference based on noisy distance to agents and updates opponents distributions
    def positionDistanceInfer(self, agentIndex, gameState=None):
        if not gameState:
            gameState=self.getCurrentObservation()

        myPos = self.getMyPos(gameState)

        noisyDistance = gameState.getAgentDistances()[agentIndex]
        beliefs= self.getmDistribs(agentIndex)
        allPossible = util.Counter()


        for p in self.legalPositions:
            trueDistance = util.manhattanDistance(p, myPos)
            if beliefs[p]==0:
                #don't need to do anything
                pass
            elif trueDistance<=SIGHT_RANGE:
                #agent would be visible if it were here, so its not here
                allPossible[p]=0
            #if this position is not on the side of the board the agent is currently on, then the agent isn't at this location
            elif self.onMySide(gameState, p) != gameState.getAgentState(agentIndex).isPacman:
                allPossible[p]=0
            else:
                allPossible[p]=beliefs[p]*gameState.getDistanceProb(trueDistance, noisyDistance)

        allPossible.normalize()
        self.setDistrib(agentIndex, allPossible)

    #does inference based on where the agent could move to and updates opponents distributions
    def positionMoveInfer(self, agentIndex, gameState=None, beliefs=None):
        if not gameState:
            gameState=self.getCurrentObservation()
        if not beliefs:
            beliefs= self.getmDistribs(agentIndex)
        #myState=gameState.getAgentState(self.index)
        myPos= self.getMyPos(gameState)

        possiblePositions = util.Counter()

        for pos in self.legalPositions:
            #if the distance is less than SIGHT_RANGE, don't need to do inference on this position, bc we know the agent isn't there
            if beliefs[pos] > 0:
                newPosDist = self.getPositionDistribution(agentIndex, pos, gameState)
                for position, prob in newPosDist.items():
                    possiblePositions[position] += prob * beliefs[pos]

        possiblePositions.normalize()
        return possiblePositions


    #returns a probability distribution for the agents subsequent position, given that it is at curPos
    def getPositionDistribution(self, agentIndex, curPos, gameState=None):
        if not gameState:
            gameState=self.getCurrentObservation()
        neighbors = game.Actions.getLegalNeighbors(curPos, gameState.getWalls())
        probs={}
        #currently assumes agressively moves towards closest "objective" (food or pacman) with probability .8
        objectives=self.data.defendFoodGrid[-1].asList()
        for i in self.getTeam(gameState):
            if gameState.getAgentState(i).isPacman:
                objectives.append(gameState.getAgentPosition(i))

        minDist=self.getMazeDistance(neighbors[0], objectives[0])
        bestNeighbor=neighbors[0]
        #find the neighbor that is closest to an objective
        for obj in objectives:
            for neighbor in neighbors:
                if self.getMazeDistance(obj, neighbor)<minDist:
                    bestNeighbor=neighbor
        defProb=.8
        otherProbs=(1-defProb)/(len(neighbors)-1)
        #set the probability we move to a neighbor that is not bestNeighbor to the correct value
        for n in neighbors:
            probs[n]=otherProbs
        probs[bestNeighbor]=defProb

        return probs

    #compares the most recent food log to the one before it, looking for any food that disappeared
    def checkFood(self):
        if len(self.data.defendFoodGrid) < 2:
            return False
        prevFood=self.data.defendFoodGrid[-2]
        currFood=self.data.defendFoodGrid[-1]
        halfway = currFood.width / 2
        #copied from halfgrid
        if self.red:
            xrange = range(halfway)
        else:
            xrange = range(halfway, currFood.width)
        #TODO: can check numCarrying of previous agent to see if it changed, only do this check if it ate food
        for y in range(currFood.height):
            for x in xrange:
                if prevFood[x][y] and not currFood[x][y]:
                    #food has been eaten in the past move
                    self._setKnownPosDist(self.getPrevPlayer(), (x,y))
                    return True

    #checks if we can see either of the opponents, if so, updates their belief state and doesn't do inference
    #if not, does inference
    def updatePosDist(self, gameState=None):
        if not gameState:
            gameState=self.getCurrentObservation()
        for i in self.getOpponents(gameState):
            if gameState.getAgentPosition(i): #can observe the given agent

                self._setKnownPosDist(i, gameState.getAgentPosition(i))
            #Only do move infer on the agent right before the current agent, as both agents haven't moved since last call
            #(if this is agent 3, agent 2 just moved, but agent 4 has not moved since agent 1 did inference.
            elif self.getPrevPlayer()==i: #i is the previous agent
                if self.index==0 and self.getPreviousObservation()==None: #this is the first move, don't do inference
                    pass
                else:
                    #check if any food was eaten. If so, don't do inference. if not, do inference
                    if not self.checkFood():
                        self.positionMoveInfer(i)
                        #positionDistanceInfer returns the new distribution, so update the saved distribution
                        self.setDistrib(i, self.positionMoveInfer(i))
            else:
                #do inference based on distance
                self.positionDistanceInfer(i)

