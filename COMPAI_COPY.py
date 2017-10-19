# baselineTeam.py
# ---------------
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


# baselineTeam.py
# ---------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

from captureAgents import CaptureAgent
import distanceCalculator
import random, time, util, sys
from game import Directions
import game
from util import nearestPoint

#################
# Team creation #
#################

import sys

sys.path.append("teams/<COMPAI>/")


def createTeam(firstIndex, secondIndex, isRed,
               first='Attacker', second='Defender'):
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
    return [eval(first)(firstIndex), eval(second)(secondIndex)]


##########
# Agents #
##########



class Actions():
    """
    A base class for all the actions that can be used by both attacker and defender.
    """

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
            return successor

    def evaluate(self, gameState, action):
        """
        Computes a linear combination of features and feature weights
        """
        features = self.getFeatures(gameState, action)
        weights = self.getWeights(gameState, action)
        return features * weights

    def getFeatures(self, gameState, action):
        """
        Returns a counter of features for the state
        """
        features = util.Counter()
        successor = self.getSuccessor(gameState, action)
        features['successorScore'] = self.agent.getScore(successor)
        return features

    def getWeights(self, gameState, action):
        """
        Normally, weights do not depend on the gamestate.  They can be either
        a counter or a dictionary.
        """
        return {'successorScore': 1.0}


class getOffensiveActions(Actions):
    def __init__(self, agent, index, gameState):
        self.agent = agent
        self.index = index
        self.agent.distancer.getMazeDistances()
        self.retreat = False
        self.numEnemyFood = "+inf"
        self.counter = 0;

        if self.agent.red:
            boundary = (gameState.data.layout.width - 2) / 2
        else:
            boundary = ((gameState.data.layout.width - 2) / 2) + 1
        self.boundary = []
        for i in range(1, gameState.data.layout.height - 1):
            if not gameState.hasWall(boundary, i):
                self.boundary.append((boundary, i))


        self.patrolSpot = []
        while len(self.patrolSpot) > (gameState.data.layout.height - 2) / 2:
            self.patrolSpot.pop(0)
            self.patrolSpot.pop(len(self.patrolSpot) - 1)
            # Update probabilities to each patrol point.

    def getFeatures(self, gameState, action):
        """
        Get features used for state evaluation.
        """
        features = util.Counter()
        successor = self.getSuccessor(gameState, action)

        # Compute score from successor state
        features['successorScore'] = self.agent.getScore(successor)
        # get current position of the agent

        CurrentPosition = successor.getAgentState(self.index).getPosition()

        # Compute the distance to the nearest boundary
        boundaryMin = 1000000
        for i in range(len(self.boundary)):
            disBoundary = self.agent.getMazeDistance(CurrentPosition, self.boundary[i])
            if (disBoundary < boundaryMin):
                boundaryMin = disBoundary
        features['returned'] = boundaryMin

        features['carrying'] = successor.getAgentState(self.index).numCarrying
        # Compute distance to the nearest food
        foodList = self.agent.getFood(successor).asList()
        if len(foodList) > 0 :
            minFoodDistance = 99999
            for food in foodList:
                distance = self.agent.getMazeDistance(CurrentPosition, food)
                if (distance < minFoodDistance):
                    minFoodDistance = distance
            features['distanceToFood'] = minFoodDistance

        # Compute distance to the nearest capsule
        capsuleList = self.agent.getCapsules(successor)
        if len(capsuleList) > 0:
            minCapsuleDistance = 99999
            for c in capsuleList:
                distance = self.agent.getMazeDistance(CurrentPosition, c)
                if distance < minCapsuleDistance:
                    minCapsuleDistance = distance
            features['distanceToCapsule'] = minCapsuleDistance
        else:
            features['distanceToCapsule'] = 0

        # Compute distance to closest ghost
        opponentsState = []
        for i in self.agent.getOpponents(successor):
            opponentsState.append(successor.getAgentState(i))
        visible = filter(lambda x: not x.isPacman and x.getPosition() != None, opponentsState)
        if len(visible) > 0:
            positions = [agent.getPosition() for agent in visible]
            closest = min(positions, key=lambda x: self.agent.getMazeDistance(CurrentPosition, x))
            closestDist = self.agent.getMazeDistance(CurrentPosition, closest)
            if closestDist <= 5:
                # print(CurrentPosition,closest,closestDist)
                features['GhostDistance'] = closestDist

        else:
            probDist = []
            for i in self.agent.getOpponents(successor):
                probDist.append(successor.getAgentDistances()[i])
            features['GhostDistance'] = min(probDist)

        # Attacker only try to kill the enemy if : itself is ghost form and the distance between him and the ghost is less than 4
        enemiesPacMan = [successor.getAgentState(i) for i in self.agent.getOpponents(successor)]
        Range = filter(lambda x: x.isPacman and x.getPosition() != None, enemiesPacMan)
        if len(Range) > 0:
            positions = [agent.getPosition() for agent in Range]
            closest = min(positions, key=lambda x: self.agent.getMazeDistance(CurrentPosition, x))
            closestDist = self.agent.getMazeDistance(CurrentPosition, closest)
            if closestDist < 4:
                # print(CurrentPosition,closest,closestDist)
                features['distanceToEnemiesPacMan'] = closestDist
        else:
            features['distanceToEnemiesPacMan'] = 0

        features['eaten']=1
        return features

    def getWeights(self, gameState, action):
        """
        Get weights for the features used in the evaluation.
        """

        # If opponent is scared, the agent should not care about GhostDistance
        successor = self.getSuccessor(gameState, action)
        numOfCarrying = successor.getAgentState(self.index).numCarrying
        opponents = [successor.getAgentState(i) for i in self.agent.getOpponents(successor)]
        curPos=gameState.getAgentState(self.index).getPosition()
        sucPos=successor.getAgentState(self.index).getPosition()
        visible = filter(lambda x: not x.isPacman and x.getPosition() != None, opponents)
        if len(visible) > 0:
            for agent in visible:
                if self.agent.getMazeDistance(sucPos,agent.getPosition())<=5:
                    if agent.scaredTimer > 0:
                        if agent.scaredTimer > 12:
                            print("aaa",10-3*numOfCarrying)
                            return {'successorScore': 110, 'distanceToFood': -10, 'distanceToEnemiesPacMan': 0,
                                    'GhostDistance': 0, 'distanceToCapsule': 0, 'returned': 10-3*numOfCarrying, 'carrying': 350,'eaten':0}

                        elif 6 < agent.scaredTimer < 12 :
                            print("bbb")
                            return {'successorScore': 110++5*numOfCarrying, 'distanceToFood': -5, 'distanceToEnemiesPacMan': 0,
                                    'GhostDistance': 0, 'distanceToCapsule': -10, 'returned': -5-4*numOfCarrying,
                                    'carrying': 200,'eaten':0}

                    # Visible and not scared
                    else:
                        #get eaten
                        if self.agent.getMazeDistance(sucPos,curPos)>1:
                            return {'successorScore': 110, 'distanceToFood': -5, 'distanceToEnemiesPacMan': 0,
                                    'GhostDistance':35 , 'distanceToCapsule': -10, 'returned': -1-4*numOfCarrying,'carrying': 0,'eaten':-1000000
                                    }
                        return {'successorScore': 110, 'distanceToFood': -5, 'distanceToEnemiesPacMan': 0,
                                'GhostDistance': 35, 'distanceToCapsule': -10, 'returned': -1 - 4 * numOfCarrying,
                                'carrying': 0, 'eaten': 0
                                }


        # If I am not PacMan the enemy is a pacMan, I can try to eliminate him
        # Attacker only try to defence if it is close to it (less than 4 steps)
        # enemiesPacMan = [successor.getAgentState(i) for i in self.agent.getOpponents(successor)]
        # Range = filter(lambda x: x.isPacman and x.getPosition() != None, enemiesPacMan)
        # if len(Range) > 0 and not gameState.getAgentState(self.index).isPacman:
        #     return {'successorScore': 0, 'distanceToFood': -1, 'distanceToEnemiesPacMan': -8,
        #             'distanceToCapsule': 0, 'GhostDistance': 0,
        #             'returned': 0, 'carrying': 10}

        # Did not see anything
        self.counter += 1
        #print("Counter ",self.counter)
        return {'successorScore': 1000+numOfCarrying*3.5, 'distanceToFood': -7, 'GhostDistance': 0, 'distanceToEnemiesPacMan': 0,
                'distanceToCapsule': -5, 'returned': 5-numOfCarrying*3, 'carrying': 350}

    def allSimulation(self, depth, gameState, decay):
        new_state = gameState.deepCopy()
        if depth == 0:
            result_list = []
            actions = new_state.getLegalActions(self.index)
            actions.remove(Directions.STOP)

            reversed_direction = Directions.REVERSE[new_state.getAgentState(self.index).configuration.direction]
            if reversed_direction in actions and len(actions) > 1:
                actions.remove(reversed_direction)
            a = random.choice(actions)
            next_state = new_state.generateSuccessor(self.index, a)
            result_list.append(self.evaluate(next_state, Directions.STOP))
            return max(result_list)

        # Get valid actions
        result_list = []
        actions = new_state.getLegalActions(self.index)
        current_direction = new_state.getAgentState(self.index).configuration.direction
        # The agent should not use the reverse direction during simulation

        reversed_direction = Directions.REVERSE[current_direction]
        if reversed_direction in actions and len(actions) > 1:
            actions.remove(reversed_direction)

        # Randomly chooses a valid action
        for a in actions:
            # Compute new state and update depth
            next_state = new_state.generateSuccessor(self.index, a)
            result_list.append(
                self.evaluate(next_state, Directions.STOP) + decay * self.allSimulation(depth - 1, next_state, decay))
        return max(result_list)

    def MTCS(self, depth, gameState, decay):

        new_state = gameState.deepCopy()
        value = self.evaluate(new_state, Directions.STOP)
        decay_index = 1
        while depth > 0:

            # Get valid actions
            actions = new_state.getLegalActions(self.index)
            # The agent should not stay put in the simulation
            # actions.remove(Directions.STOP)
            current_direction = new_state.getAgentState(self.agent.index).configuration.direction
            # The agent should not use the reverse direction during simulation

            reversed_direction = Directions.REVERSE[new_state.getAgentState(self.agent.index).configuration.direction]
            if reversed_direction in actions and len(actions) > 1:
                actions.remove(reversed_direction)
            # Randomly chooses a valid action
            a = random.choice(actions)
            # Compute new state and update depth
            new_state = new_state.generateSuccessor(self.agent.index, a)
            value = value + decay ** decay_index * self.evaluate(new_state, Directions.STOP)
            depth -= 1
            decay_index += 1
        # Evaluate the final simulation state
        return value


    def chooseAction(self, gameState):
        start = time.time()

        # Get valid actions. Randomly choose a valid one out of the best (if best is more than one)
        actions = gameState.getLegalActions(self.agent.index)
        actions.remove(Directions.STOP)
        possibleValue = []

        for a in actions:
            NextState = gameState.generateSuccessor(self.agent.index, a)
            value = self.allSimulation(2, NextState, 0.7)
            possibleValue .append(value)

        # Choose randomly from possible (very unlikely) best
        bestAction = max(possibleValue)
        possibleChoice = filter(lambda x: x[0] == bestAction, zip(possibleValue, actions))
        #print 'eval time for offensive agent %d: %.4f' % (self.agent.index, time.time() - start)
        return random.choice(possibleChoice)[1]



class getDefensiveActions(Actions):
    # Load the denfensive information
    def __init__(self, agent, index, gameState):
        self.index = index
        self.agent = agent
        self.defenderList = {}
        if self.agent.red:
            middle = (gameState.data.layout.width - 2) / 2
        else:
            middle = ((gameState.data.layout.width - 2) / 2) + 1
        self.noWall = []
        for i in range(1, gameState.data.layout.height - 1):
            if not gameState.hasWall(middle, i):
                self.noWall.append((middle, i))

        self.target = None
        self.lastObservedFood = None
        # Remove some positions. The agent do not need to patrol
        # all positions in the central area.
        while len(self.noWall) > (gameState.data.layout.height - 2) / 2:
            self.noWall.pop(0)
            self.noWall.pop(len(self.noWall) - 1)
        # Update probabilities to each patrol point.
        self.DefendingProbability(gameState)

    def DefendingProbability(self, gameState):

        total = 0

        # Get the minimum distance from the food to our
        # patrol points.
        for position in self.noWall:
            closestFoodDistance = 99999
            foodList = self.agent.getFoodYouAreDefending(gameState).asList()
            for food in foodList:
                dist = self.agent.getMazeDistance(position, food)
                if dist < closestFoodDistance:
                    closestFoodDistance = dist

            if closestFoodDistance == 0:
                closestFoodDistance = 1
            self.defenderList[position] = 1.0 / float(closestFoodDistance)
            total += self.defenderList[position]

        # Normalize the value used as probability.
        if total == 0:
            total = 1
        for x in self.defenderList.keys():
            self.defenderList[x] = float(self.defenderList[x]) / float(total)

    def selectPatrolTarget(self):

        rand = random.random()
        sum = 0.0
        for x in self.defenderList.keys():
            sum += self.defenderList[x]
            if rand < sum:
                return x


    def chooseAction(self, gameState):

        # start = time.time()
        #  Patrol probabilities.

        DefendingList = self.agent.getFoodYouAreDefending(gameState).asList()
        if self.lastObservedFood and len(self.lastObservedFood) != len(DefendingList):
            self.DefendingProbability(gameState)

        CurrentPosition = gameState.getAgentPosition(self.index)
        if CurrentPosition == self.target:
            self.target = None

        # Visible enemy , keep chasing.
        opponentsState = []
        for i in self.agent.getOpponents(gameState):
            opponentsState.append(gameState.getAgentState(i))
        visible = filter(lambda x:x.isPacman and x.getPosition() != None, opponentsState)
        if len(visible)>0:
            positions = []
            for invader in visible:
                positions.append(invader.getPosition())

            self.target = min(positions, key=lambda x: self.agent.getMazeDistance(CurrentPosition, x))

        # If we can't see an invader, but our pacdots were eaten,
        # we will check the position where the pacdot disappeared.
        elif self.lastObservedFood != None:
            eaten = set(self.lastObservedFood) - set(self.agent.getFoodYouAreDefending(gameState).asList())
            if len(eaten)>0:
                self.target = eaten.pop()

        # Update the distribution.
        self.lastObservedFood = self.agent.getFoodYouAreDefending(gameState).asList()

        # If we have only a few pacdots, let's walk among them.
        if self.target == None and len(self.agent.getFoodYouAreDefending(gameState).asList()) <= 4:
            food = self.agent.getFoodYouAreDefending(gameState).asList() \
                   + self.agent.getCapsulesYouAreDefending(gameState)
            self.target = random.choice(food)

        # Random patrolling
        elif self.target == None:
            self.target = self.selectPatrolTarget()

        # Choose action. We will take the action that brings us
        # closer to the target. However, we will never stay put
        # and we will never invade the enemy side.

        actions = gameState.getLegalActions(self.index)
        feasible = []
        fvalues = []
        for a in actions:
            new_state = gameState.generateSuccessor(self.index, a)
            if not a == Directions.STOP and not new_state.getAgentState(self.index).isPacman:
                newPosition = new_state.getAgentPosition(self.index)
                feasible.append(a)
                fvalues.append(self.agent.getMazeDistance(newPosition, self.target))

        # Randomly chooses between ties.
        best = min(fvalues)
        ties = filter(lambda x: x[0] == best, zip(fvalues, feasible))

        # print 'eval time for defender agent %d: %.4f' % (self.index, time.time() - start)
        return random.choice(ties)[1]


class Attacker(CaptureAgent):
    def __init__(self, index):
        CaptureAgent.__init__(self, index)

    def registerInitialState(self, gameState):
        CaptureAgent.registerInitialState(self, gameState)
        self.DefenceStatus = getDefensiveActions(self, self.index, gameState)
        self.OffenceStatus = getOffensiveActions(self, self.index, gameState)

    def chooseAction(self, gameState):
        self.enemies = self.getOpponents(gameState)

        if  self.getScore(gameState) > 15:
            return self.DefenceStatus.chooseAction(gameState)
        else:
            return self.OffenceStatus.chooseAction(gameState)


class Defender(CaptureAgent):
    def __init__(self, index):
        CaptureAgent.__init__(self, index)

    def registerInitialState(self, gameState):
        CaptureAgent.registerInitialState(self, gameState)
        self.DefenceStatus = getDefensiveActions(self, self.index, gameState)
        self.OffenceStatus = getOffensiveActions(self, self.index, gameState)

    def chooseAction(self, gameState):
        self.enemies = self.getOpponents(gameState)
        # if numInvaders == 0 and self.getScore(gameState) < 10:
        #     return self.OffenceStatus.chooseAction(gameState)
        # else:
        #     print(self.DefenceStatus.target, "Target is ..........")
        return self.DefenceStatus.chooseAction(gameState)



