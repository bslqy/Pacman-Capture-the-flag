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

sys.path.append("teams/<clearlove3>/")


def createTeam(firstIndex, secondIndex, isRed,
               first='OffensiveReflexAgent', second='DefensiveReflexAgent'):
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

class ReflexCaptureAgent(CaptureAgent):
    """
    A base class for reflex agents that chooses score-maximizing actions
    """

    def registerInitialState(self, gameState):
        self.start = gameState.getAgentPosition(self.index)
        CaptureAgent.registerInitialState(self, gameState)

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
        features['successorScore'] = self.getScore(successor)
        return features

    def getWeights(self, gameState, action):
        """
        Normally, weights do not depend on the gamestate.  They can be either
        a counter or a dictionary.
        """
        return {'successorScore': 1.0}


class OffensiveReflexAgent(ReflexCaptureAgent):
    """
    A reflex agent that seeks food. This is an agent
    we give you to get an idea of what an offensive agent might look like,
    but it is by no means the best or only way to build an offensive agent.
    """

    def getFeatures(self, gameState, action):
        """
        Get features used for state evaluation.
        """
        features = util.Counter()
        successor = self.getSuccessor(gameState,action)
        # Compute score from successor state
        features['successorScore'] = self.getScore(successor)-self.getScore(gameState)
        # get current position of the agent

        myPos = gameState.getAgentState(self.index).getPosition()

        #get next position of the agent
        sucPos = successor.getAgentState(self.index).getPosition()
        # Compute distance to the nearest food
        foodList = self.getFood(gameState).asList()
        if len(foodList) > 0:
            minDistance = min([self.getMazeDistance(myPos, food) for food in foodList])
            sucFoodDistance=min([self.getMazeDistance(sucPos, food) for food in foodList])
            features['towardsFood'] =minDistance-sucFoodDistance

        # Compute distance to closest ghost
        features['deadends']=0
        features['towardsGhost']=0
        enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
        inRange = filter(lambda x: not x.isPacman and x.getPosition() != None and self.getMazeDistance(myPos,x.getPosition())<5 and x.scaredTimer<5, enemies)
        if len(inRange) > 0:
            positions = [agent.getPosition() for agent in inRange]
            closest = min(positions, key=lambda x: self.getMazeDistance(myPos, x))
            ghostDis = self.getMazeDistance(myPos, closest)
            sucGhostDis=self.getMazeDistance(sucPos,closest)
            #going to deadends
            if self.deadEnds.has_key((myPos, action)) and self.deadEnds[(myPos, action)] * 2 > ghostDis:
                features['deadends'] =1
            #the ghost is about to eat pacman
            elif ghostDis<3:
                runaway=sucGhostDis-ghostDis
                #you are eaten
                if runaway>10:
                    features['towardsGhost']=-100
                else:
                    features['towardsGhost']=runaway
                capsuleList = self.getCapsules(gameState)
                if len(capsuleList) > 0:
                    minCapDistance = min([self.getMazeDistance(myPos, c) for c in capsuleList])
                    sucCapDistance=min([self.getMazeDistance(sucPos, c) for c in capsuleList])
                    features['towardsCapsule'] = minCapDistance-sucCapDistance
                else:
                    features['towardsCapsule'] = 0

        # Compute distance to the nearest capsule

        """
        capsuleList = self.getCapsules(successor)
        if len(capsuleList) > 0:
            minDistance = min([self.getMazeDistance(myPos, c) for c in capsuleList])
            features['distanceToCapsule'] = minDistance
        else:
            features['distanceToCapsule'] = 0
        """

        # features['distanceToMid'] = min([self.cap.distancer.getDistance(myPos, i)
        #                                  for i in self.noWallSpots])

        # Compute the distance to the nearest boundary

        boundaryMin = 1000000
        for i in range(len(self.boundary)):
            disBoundary = self.getMazeDistance(myPos, self.boundary[i])
            if (disBoundary < boundaryMin):
                boundaryMin = disBoundary
        features['returned'] = boundaryMin

        features['carrying'] = successor.getAgentState(self.index).numCarrying

        return features

    def getWeights(self, gameState, action):
        """
        Get weights for the features used in the evaluation.
        """

        return {'successorScore': 500, 'towardsFood': 20, 'towardsGhost': 500, 'towardsCapsule':300,'deadends': -2000,}

    def allSimulation(self, depth, gameState, decay):
        new_state = gameState.deepCopy()
        if depth == 0:
            result_list = []
            actions = new_state.getLegalActions(self.index)
            actions.remove(Directions.STOP)
            """
            reversed_direction = Directions.REVERSE[new_state.getAgentState(self.index).configuration.direction]
            if reversed_direction in actions and len(actions) > 1:
                actions.remove(reversed_direction)"""
            result_list.append(max(self.evaluate(new_state, a) for a in actions))
            return max(result_list)

        # Get valid actions
        result_list = []
        actions = new_state.getLegalActions(self.index)
        actions.remove(Directions.STOP)
        # The agent should not stay put in the simulation
        # actions.remove(Directions.STOP)
        # current_direction = new_state.getAgentState(self.index).configuration.direction
        # The agent should not use the reverse direction during simulation
        """
        reversed_direction = Directions.REVERSE[current_direction]
        if reversed_direction in actions and len(actions) > 1:
            actions.remove(reversed_direction)
            """
        # Randomly chooses a valid action
        for a in actions:
            # Compute new state and update depth
            next_state = new_state.generateSuccessor(self.index, a)
            result_list.append(
                self.evaluate(new_state, a) + decay * self.allSimulation(depth - 1, next_state, decay))
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
            current_direction = new_state.getAgentState(self.index).configuration.direction
            # The agent should not use the reverse direction during simulation

            reversed_direction = Directions.REVERSE[new_state.getAgentState(self.index).configuration.direction]
            if reversed_direction in actions and len(actions) > 1:
                actions.remove(reversed_direction)
            # Randomly chooses a valid action
            a = random.choice(actions)
            # Compute new state and update depth
            new_state = new_state.generateSuccessor(self.index, a)
            value = value + decay ** decay_index * self.evaluate(new_state, Directions.STOP)
            depth -= 1
            decay_index += 0.5
        # Evaluate the final simulation state
        return value

    def __init__(self, index):
        CaptureAgent.__init__(self, index)
        # Variables used to verify if the agent os locked
        # self.numEnemyFood = "+inf"
        # self.inactiveTime = 0

        # Implemente este metodo para pre-processamento (15s max).

    def registerInitialState(self, gameState):
        CaptureAgent.registerInitialState(self, gameState)
        # get the deadends of the map
        self.deadEnds = {}
        # get the feasible position of the map
        self.feasible = []
        for i in range(1, gameState.data.layout.height - 1):
            for j in range(1, gameState.data.layout.width - 1):
                if not gameState.hasWall(j, i):
                    self.feasible.append((j, i))
        # store the crossroads met in the travel
        crossRoad = util.Queue()

        currentState = gameState
        # the entrance of the deadend
        entPos = currentState.getAgentPosition(self.index)
        entDirection = currentState.getAgentState(self.index).configuration.direction
        actions = currentState.getLegalActions(self.index)
        actions.remove(Directions.STOP)
        for a in actions:
            crossRoad.push((currentState,a))
        # if there is still some positions unexplored
        while not crossRoad.isEmpty():
            # if it is not a crossroad nor a deadend

            (entState,entDirection) = crossRoad.pop()
            depth = 0
            entPos = entState.getAgentState(self.index).getPosition()
            currentState=entState.generateSuccessor(self.index,entDirection)
            while True:
                # get current position

                currentPos = currentState.getAgentState(self.index).getPosition()
                # get next actions
                actions = currentState.getLegalActions(self.index)
                actions.remove(Directions.STOP)
                currentDirection = currentState.getAgentState(self.index).configuration.direction
                if currentPos not in self.feasible:
                    break
                self.feasible.remove(currentPos)
                if Directions.REVERSE[currentDirection] in actions:
                    actions.remove(Directions.REVERSE[currentDirection])

                # deadend
                if len(actions) == 0:
                    self.deadEnds[(entPos, entDirection)] = depth + 1
                    break

                # there is only one direction to move
                elif len(actions) == 1:
                    depth = depth + 1
                    # generate next state
                    currentState = currentState.generateSuccessor(self.index, actions[0])
                # meet crossroad
                else:
                    # get the successors
                    for a in actions:
                        crossRoad.push((currentState,a))

                    break

        for i in self.deadEnds.keys():
            print(i, self.deadEnds[i])

        self.distancer.getMazeDistances()
        if self.red:
            centralX = (gameState.data.layout.width - 2) / 2
        else:
            centralX = ((gameState.data.layout.width - 2) / 2) + 1
        self.boundary = []
        for i in range(1, gameState.data.layout.height - 1):
            if not gameState.hasWall(centralX, i):
                self.boundary.append((centralX, i))

    def chooseAction(self, gameState):
        # You can profile your evaluation time by uncommenting these lines
        start = time.time()

        # Get valid actions. Staying put is almost never a good choice, so
        # the agent will ignore this action.
        actions = gameState.getLegalActions(self.index)
        actions.remove(Directions.STOP)
        fvalues = []
        for a in actions:
            newState = gameState.generateSuccessor(self.index, a)
            value = self.evaluate(gameState,a)
            value += self.allSimulation(3, newState, 0.5)
            fvalues.append(value)

        best = max(fvalues)
        ties = filter(lambda x: x[0] == best, zip(fvalues, actions))
        toPlay = random.choice(ties)[1]
        print("best:", best, toPlay)
        print 'eval time for offensive agent %d: %.4f' % (self.index, time.time() - start)
        return toPlay


class DefensiveReflexAgent(ReflexCaptureAgent):
    """
    A reflex agent that keeps its side Pacman-free. Again,
    this is to give you an idea of what a defensive agent
    could be like.  It is not the best or only way to make
    such an agent.
    """

    def __init__(self, index):
        CaptureAgent.__init__(self, index)
        self.target = None
        self.lastObservedFood = None
        # This variable will store our patrol points and
        # the agent probability to select a point as target.
        self.patrolDict = {}

    def distFoodToPatrol(self, gameState):
        """
        This method calculates the minimum distance from our patrol
        points to our pacdots. The inverse of this distance will
        be used as the probability to select the patrol point as
        target.
        """
        food = self.getFoodYouAreDefending(gameState).asList()
        total = 0

        # Get the minimum distance from the food to our
        # patrol points.
        for position in self.noWallSpots:
            closestFoodDist = "+inf"
            for foodPos in food:
                dist = self.getMazeDistance(position, foodPos)
                if dist < closestFoodDist:
                    closestFoodDist = dist
            # We can't divide by 0!
            if closestFoodDist == 0:
                closestFoodDist = 1
            self.patrolDict[position] = 1.0 / float(closestFoodDist)
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

    # Implemente este metodo para pre-processamento (15s max).
    def registerInitialState(self, gameState):
        CaptureAgent.registerInitialState(self, gameState)
        self.distancer.getMazeDistances()

        # Compute central positions without walls from map layout.
        # The defender will walk among these positions to defend
        # its territory.
        if self.red:
            centralX = (gameState.data.layout.width - 2) / 2
        else:
            centralX = ((gameState.data.layout.width - 2) / 2) + 1
        self.noWallSpots = []
        for i in range(1, gameState.data.layout.height - 1):
            if not gameState.hasWall(centralX, i):
                self.noWallSpots.append((centralX, i))
        # Remove some positions. The agent do not need to patrol
        # all positions in the central area.
        while len(self.noWallSpots) > (gameState.data.layout.height - 2) / 2:
            self.noWallSpots.pop(0)
            self.noWallSpots.pop(len(self.noWallSpots) - 1)
        # Update probabilities to each patrol point.
        self.distFoodToPatrol(gameState)

    # Implemente este metodo para controlar o agente (1s max).
    def chooseAction(self, gameState):
        # You can profile your evaluation time by uncommenting these lines
        # start = time.time()

        # If some of our food was eaten, we need to update
        # our patrol points probabilities.
        if self.lastObservedFood and len(self.lastObservedFood) != len(self.getFoodYouAreDefending(gameState).asList()):
            self.distFoodToPatrol(gameState)

        mypos = gameState.getAgentPosition(self.index)
        if mypos == self.target:
            self.target = None

        # If we can see an invader, we go after him.
        x = self.getOpponents(gameState)
        enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
        invaders = filter(lambda x: x.isPacman and x.getPosition() != None, enemies)
        if len(invaders) > 0:
            positions = [agent.getPosition() for agent in invaders]
            self.target = min(positions, key=lambda x: self.getMazeDistance(mypos, x))
        # If we can't see an invader, but our pacdots were eaten,
        # we will check the position where the pacdot disappeared.
        elif self.lastObservedFood != None:
            eaten = set(self.lastObservedFood) - set(self.getFoodYouAreDefending(gameState).asList())
            if len(eaten) > 0:
                self.target = eaten.pop()

        # Update the agent memory about our pacdots.
        self.lastObservedFood = self.getFoodYouAreDefending(gameState).asList()

        # No enemy in sight, and our pacdots are not disappearing.
        # If we have only a few pacdots, let's walk among them.
        if self.target == None and len(self.getFoodYouAreDefending(gameState).asList()) <= 4:
            food = self.getFoodYouAreDefending(gameState).asList() \
                   + self.getCapsulesYouAreDefending(gameState)
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
            if not new_state.getAgentState(self.index).isPacman and not a == Directions.STOP:
                newpos = new_state.getAgentPosition(self.index)
                goodActions.append(a)
                fvalues.append(self.getMazeDistance(newpos, self.target))

        # Randomly chooses between ties.
        best = min(fvalues)
        ties = filter(lambda x: x[0] == best, zip(fvalues, goodActions))

        # print 'eval time for defender agent %d: %.4f' % (self.index, time.time() - start)
        return random.choice(ties)[1]

    """
    def getFeatures(self, gameState, action):
        features = util.Counter()
        successor = self.getSuccessor(gameState, action)

        myState = successor.getAgentState(self.index)
        myPos = myState.getPosition()

        # Computes whether we're on defense (1) or offense (0)
        features['onDefense'] = 1
        if myState.isPacman: features['onDefense'] = 0

        # Computes distance to invaders we can see
        enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
        invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]
        features['numInvaders'] = len(invaders)
        if len(invaders) > 0:
            dists = [self.getMazeDistance(myPos, a.getPosition()) for a in invaders]
            features['invaderDistance'] = min(dists)

        if action == Directions.STOP: features['stop'] = 1
        rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
        if action == rev: features['reverse'] = 1

        return features

    def getWeights(self, gameState, action):
        return {'numInvaders': -1000, 'onDefense': 100, 'invaderDistance': -10, 'stop': -100, 'reverse': -2}
"""