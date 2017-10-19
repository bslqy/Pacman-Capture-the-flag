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

        if self.agent.red:
            boundary = (gameState.data.layout.width - 2) / 2
        else:
            boundary = ((gameState.data.layout.width - 2) / 2) + 1
        self.boundary = []
        for i in range(1, gameState.data.layout.height - 1):
            if not gameState.hasWall(boundary, i):
                self.boundary.append((boundary, i))

        # Remove some positions. The agent do not need to patrol
        # all positions in the central area.
        self.noWallSpots = []
        while len(self.noWallSpots) > (gameState.data.layout.height - 2) / 2:
            self.noWallSpots.pop(0)
            self.noWallSpots.pop(len(self.noWallSpots) - 1)
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

        # Compute if is pacman
        features['isPacman'] = 1 if successor.getAgentState(self.index).isPacman else 0

        return features

    def getWeights(self, gameState, action):
        """
        Get weights for the features used in the evaluation.
        """

        # If opponent is scared, the agent should not care about GhostDistance
        successor = self.getSuccessor(gameState, action)
        numOfFood = len(self.agent.getFood(successor).asList())
        numOfCarrying = successor.getAgentState(self.index).numCarrying
        CurrentPosition = successor.getAgentState(self.index).getPosition()
        #myself = successor.getAgentState.(self.index).isPacman
        opponents = [successor.getAgentState(i) for i in self.agent.getOpponents(successor)]
        visible = filter(lambda x: not x.isPacman and x.getPosition() != None, opponents)
        if len(visible) > 0:
            for agent in visible:
                if agent.scaredTimer > 0:
                    if agent.scaredTimer > 12:
                        return {'successorScore': 110, 'distanceToFood': -10, 'distanceToEnemiesPacMan': 0,
                                'GhostDistance': -1, 'distanceToCapsule': 0, 'returned': 10-3*numOfCarrying, 'carrying': 350}

                    elif 6 < agent.scaredTimer < 12 :
                        return {'successorScore': 110+5*numOfCarrying, 'distanceToFood': -5, 'distanceToEnemiesPacMan': 0,
                                'GhostDistance': -15, 'distanceToCapsule': -10, 'returned': -5-4*numOfCarrying,
                                'carrying': 100}

                # Visible and not scared
                else:
                    return {'successorScore': 110, 'distanceToFood': -10, 'distanceToEnemiesPacMan': 0,
                            'GhostDistance': 20, 'distanceToCapsule': -15, 'returned': -15,
                            'carrying': 0}



                # elif gameState.getAgentState(self.index).isPacman:
                #     return {'successorScore': 30, 'distanceToFood': -3, 'distanceToEnemiesPacMan': 0,
                #             'GhostDistance': 30, 'distanceToCapsule': -5, 'returned': -10, 'carrying': 20, 'isPacman':-100}

        # if gameState.getAgentState(self.index).numCarrying >= 7:
        # return {'successorScore': 500, 'distanceToFood': 100, 'GhostDistance': 20, 'distanceToEnemiesPacMan': 0,
        #         'distanceToCapsule': -55, 'returned': -200, 'carrying': 0}

        # If I am not PacMan the enemy is a pacMan, I can try to eliminate him
        # Attacker only try to defence if it is close to it (less than 4 steps)
        enemiesPacMan = [successor.getAgentState(i) for i in self.agent.getOpponents(successor)]
        Range = filter(lambda x: x.isPacman and x.getPosition() != None, enemiesPacMan)
        if len(Range) > 0 and not successor.getAgentState(self.index).isPacman:
            return {'successorScore': 0, 'distanceToFood': -1, 'distanceToEnemiesPacMan': -8,
                    'distanceToCapsule': 0, 'GhostDistance': 0,
                    'returned': 0, 'carrying': 6}

        # Did not see anything
        #print ("No ONE !!!!!!!!!!!")
        return {'successorScore': 400+numOfCarrying*3.5, 'distanceToFood': -7, 'GhostDistance': 0, 'distanceToEnemiesPacMan': 0,
                'distanceToCapsule': -5, 'returned': 5-numOfCarrying*2.5, 'carrying': 350}

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

    def randomSimulation(self, depth, gameState, decay):
        """
        Random simulate some actions for the agent. The actions other agents can take
        are ignored, or, in other words, we consider their actions is always STOP.
        The final state from the simulation is evaluated.
        """
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

    def randomSimulation1(self, depth, gameState, decay):
        """
        Random simulate some actions for the agent. The actions other agents can take
        are ignored, or, in other words, we consider their actions is always STOP.
        The final state from the simulation is evaluated.
        """
        # depth = 0, evaluate the next step only
        new_state = gameState.deepCopy()
        decay_index = 1
        if depth == 0:
            return self.evaluate(new_state, Directions.STOP)

        # depth > 0 , evaluate recursively with decay
        else:
            # Get valid actions
            total_below = 0
            actions = new_state.getLegalActions(self.agent.index)
            # The agent should not stay put in the simulation
            # actions.remove(Directions.STOP)
            current_direction = new_state.getAgentState(self.agent.index).configuration.direction
            # The agent should not use the reverse direction during simulation
            reversed_direction = Directions.REVERSE[current_direction]
            if reversed_direction in actions and len(actions) > 1:
                actions.remove(reversed_direction)

            # Randomly chooses a valid action
            a = random.choice(actions)
            next_state = new_state.generateSuccessor(self.agent.index, a)
            total_below = total_below + self.evaluate(next_state,
                                                      Directions.STOP) + decay ** decay_index * self.randomSimulation1(
                depth - 1, next_state, decay)
        return total_below

    def chooseAction(self, gameState):
        # You can profile your evaluation time by uncommenting these lines
        start = time.time()

        # Get valid actions. Staying put is almost never a good choice, so
        # the agent will ignore this action.

        actions = gameState.getLegalActions(self.agent.index)
        actions.remove(Directions.STOP)
        fvalues = []
        for a in actions:
            new_state = gameState.generateSuccessor(self.agent.index, a)
            value = 0
            # for i in range(1, 22):
            #     value += self.randomSimulation1(2, new_state, 0.8) / 30
            # fvalues.append(value)
            value = self.allSimulation(2, new_state, 0.7)
            fvalues.append(value)

        best = max(fvalues)
        ties = filter(lambda x: x[0] == best, zip(fvalues, actions))
        print(ties)
        toPlay = random.choice(ties)[1]
        # print("best:",best,toPlay)
        print 'eval time for offensive agent %d: %.4f' % (self.agent.index, time.time() - start)
        return toPlay


class getDefensiveActions(Actions):
    # Load the denfensive information
    def __init__(self, agent, index, gameState):
        # CaptureAgent.__init__(self, index)
        self.index = index
        self.agent = agent

        # This variable will store our patrol points and
        # the agent probability to select a point as target.
        self.patrolDict = {}
        if self.agent.red:
            centralX = (gameState.data.layout.width - 2) / 2
        else:
            centralX = ((gameState.data.layout.width - 2) / 2) + 1
        self.noWall = []
        for i in range(1, gameState.data.layout.height - 1):
            if not gameState.hasWall(centralX, i):
                self.noWall.append((centralX, i))

        self.target = None
        self.lastObservedFood = None
        # Remove some positions. The agent do not need to patrol
        # all positions in the central area.
        while len(self.noWall) > (gameState.data.layout.height - 2) / 2:
            self.noWall.pop(0)
            self.noWall.pop(len(self.noWall) - 1)
        # Update probabilities to each patrol point.
        self.PatrolDistribution(gameState)

    def PatrolDistribution(self, gameState):
        """
        This method calculates the minimum distance from our patrol
        points to our pacdots. The inverse of this distance will
        be used as the probability to select the patrol point as
        target.
        """
        # food = self.agent.getFoodYouAreDefending(gameState).asList()
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

            # We can't divide by 0!
            if closestFoodDistance == 0:
                closestFoodDistance = 1
            self.patrolDict[position] = 1.0 / float(closestFoodDistance)
            total += self.patrolDict[position]
        # Normalize the value used as probability.
        if total == 0:
            total = 1
        for x in self.patrolDict.keys():
            self.patrolDict[x] = float(self.patrolDict[x]) / float(total)

    def selectPatrolTarget(self):
        """
        Select some patrol point to use as target.
        """
        rand = random.random()
        sum = 0.0
        for x in self.patrolDict.keys():
            sum += self.patrolDict[x]
            if rand < sum:
                return x

    # Implemente este metodo para controlar o agente (1s max).
    def chooseAction(self, gameState):
        # You can profile your evaluation time by uncommenting these lines
        # start = time.time()
        # our patrol points probabilities.

        DefendingList = self.agent.getFoodYouAreDefending(gameState).asList()
        if self.lastObservedFood and len(self.lastObservedFood) != len(DefendingList):
            self.PatrolDistribution(gameState)

        CurrentPosition = gameState.getAgentPosition(self.index)
        if CurrentPosition == self.target:
            self.target = None

        # If we can see an invader, we go after him.
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
        # Update the agent memory about our pacdots.
        self.lastObservedFood = self.agent.getFoodYouAreDefending(gameState).asList()


        # No enemy in sight, and our pacdots are not disappearing.
        # If we have only a few pacdots, let's walk among them.
        if self.target == None and len(self.agent.getFoodYouAreDefending(gameState).asList()) <= 4:
            food = self.agent.getFoodYouAreDefending(gameState).asList() \
                   + self.agent.getCapsulesYouAreDefending(gameState)
            self.target = random.choice(food)
        # If we have many pacdots, let's patrol the map central area.
        elif self.target == None:
            self.target = self.selectPatrolTarget()

        # Choose action. We will take the action that brings us
        # closer to the target. However, we will never stay put
        # and we will never invade the enemy side.

        actions = gameState.getLegalActions(self.index)
        goodActions = []
        fvalues = []
        for a in actions:
            new_state = gameState.generateSuccessor(self.index, a)
            if not a == Directions.STOP and not new_state.getAgentState(self.index).isPacman:
                newPosition = new_state.getAgentPosition(self.index)
                goodActions.append(a)
                fvalues.append(self.agent.getMazeDistance(newPosition, self.target))

        # Randomly chooses between ties.
        best = min(fvalues)
        ties = filter(lambda x: x[0] == best, zip(fvalues, goodActions))

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
        invaders = [a for a in self.enemies if gameState.getAgentState(a).isPacman]
        numInvaders = len(invaders)

        # Check if we have the poison active.
        scaredTimes = [gameState.getAgentState(enemy).scaredTimer for enemy in self.enemies]
        if  self.getScore(gameState) >= 13:
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
        invaders = [a for a in self.enemies if gameState.getAgentState(a).isPacman]
        numInvaders = len(invaders)

        # Check if we have the poison active.
        scaredTimes = [gameState.getAgentState(enemy).scaredTimer for enemy in self.enemies]
        # if numInvaders == 0 and self.getScore(gameState) < 10:
        #     return self.OffenceStatus.chooseAction(gameState)
        # else:
        #     print(self.DefenceStatus.target, "Target is ..........")
        return self.DefenceStatus.chooseAction(gameState)



