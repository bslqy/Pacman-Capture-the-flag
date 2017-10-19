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
        carrying=gameState.getAgentState(self.index).numCarrying
        #get next position of the agent
        sucPos = successor.getAgentState(self.index).getPosition()
        # Compute distance to the nearest food
        foodList = self.getFood(gameState).asList()
        if len(foodList) > 2:
            minDistance = min([self.getMazeDistance(myPos, food) for food in foodList])
            sucFoodDistance=min([self.getMazeDistance(sucPos, food) for food in foodList])
            features['towardsFood'] =minDistance-sucFoodDistance
        #compute the distance to boundary
        curBound = min(self.getMazeDistance(myPos, b) for b in self.boundary)
        sucBound = min(self.getMazeDistance(sucPos, b) for b in self.boundary)
        features['towardsBound'] = (curBound - sucBound) * carrying
        # Compute distance to closest ghost
        features['deadends']=0
        features['towardsGhost']=0
        features['towardsBound']=0

        enemies = [gameState.getAgentState(i) for i in self.getOpponents(successor)]
        inRange = filter(lambda x: not x.isPacman and x.getPosition() != None and self.getMazeDistance(myPos,x.getPosition())<5 and x.scaredTimer<5, enemies)
        if len(inRange) > 0:
            positions = [agent.getPosition() for agent in inRange]
            closest = min(positions, key=lambda x: self.getMazeDistance(myPos, x))
            ghostDis = self.getMazeDistance(myPos, closest)
            sucGhostDis=self.getMazeDistance(sucPos,closest)
            for agent in inRange:
                if agent.getPosition()==closest:
                    fear=agent
            #going to deadends

            if self.deadEnds.has_key((myPos, action)) and self.deadEnds[(myPos, action)] * 2 < ghostDis:
                features['deadends'] =1
            #the ghost is about to eat pacman



            features['towardsBound'] = (curBound - sucBound)*10
            runaway=sucGhostDis-ghostDis
            #you are eaten
            if fear.scaredTimer<3 and fear.scaredTimer!=0:
                if runaway>10:
                    features['towardsGhost']=100
                else:
                   features['towardsGhost']=-runaway
            else:
                if runaway>10:
                    features['towardsGhost']=-100
                else:
                   features['towardsGhost']=runaway

            capsuleList = self.getCapsules(gameState)
            if len(capsuleList) > 0:
                minCapDistance = min([self.getMazeDistance(myPos, c) for c in capsuleList])
                sucCapDistance=min([self.getMazeDistance(sucPos, c) for c in capsuleList])

                features['towardsCapsule'] = minCapDistance-sucCapDistance
                if sucPos in capsuleList:
                        features['towardsCapsule']=100
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
        # Compute distance to the nearest boundary


        return features

    def getWeights(self, gameState, action):
        """
        Get weights for the features used in the evaluation.
        """

        return {'successorScore': 100, 'towardsFood': 50, 'towardsGhost': 500, 'towardsCapsule': 500,'towardsBound':5,
                            'deadends': -2000, }

    def allSimulation(self, depth, gameState, decay):
        new_state = gameState.deepCopy()
        if depth == 0:
            result_list = []
            actions = new_state.getLegalActions(self.index)
            actions.remove(Directions.STOP)
            result_list.append(max(self.evaluate(new_state, a) for a in actions))
            return max(result_list)

        # Get valid actions
        result_list = []
        actions = new_state.getLegalActions(self.index)
        actions.remove(Directions.STOP)

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
        currState = gameState.deepCopy()
        value = 0
        decay_index = 1
        while depth > 0:

            # Get valid actions
            actions = currState.getLegalActions(self.index)
            # The agent should not stay put in the simulation
            # actions.remove(Directions.STOP)
            current_direction = currState.getAgentState(self.index).configuration.direction
            # The agent should not use the reverse direction during simulation

            reversed_direction = Directions.REVERSE[currState.getAgentState(self.index).configuration.direction]
            if reversed_direction in actions and len(actions) > 1:
                actions.remove(reversed_direction)
            # Randomly chooses a valid action
            a = random.choice(actions)
            # Compute new state and update depth
            value = value + decay ** decay_index * self.evaluate(currState, a)
            currState = currState.generateSuccessor(self.index, a)
            depth -= 1
            decay_index += 1
        # Evaluate the final simulation state
        return value

    def __init__(self, index):
        CaptureAgent.__init__(self, index)

    def registerInitialState(self, gameState):
        CaptureAgent.registerInitialState(self, gameState)
        # get the deadends of the map
        self.lastAction=Directions.STOP
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
        carrying=gameState.getAgentState(self.index).numCarrying
        actions.remove(Directions.STOP)
        fvalues = []
        for a in actions:
            newState = gameState.generateSuccessor(self.index, a)

            value = self.evaluate(gameState,a)
            value += self.allSimulation(2, newState, 0.7)
            if a==Directions.REVERSE[self.lastAction] and carrying==0:
                value=value*0.1
            print(value,a)
            fvalues.append(value)
        """
            newState = gameState.generateSuccessor(self.index, a)
            for i in range(30):
                value = self.evaluate(gameState,a)
                value += self.randomSimulation(10, newState, 0.5)
            fvalues.append(value)"""
        best = max(fvalues)
        ties = filter(lambda x: x[0] == best, zip(fvalues, actions))
        toPlay = random.choice(ties)[1]
        self.lastAction=toPlay
        print(toPlay)
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
        self.previousFood = None
        # This variable will store our patrol points and
        # the agent probability to select a point as target.

    def DefendingProbability(self, gameState):
        """
        This method calculates the minimum distance from our patrol
        points to our pacdots. The inverse of this distance will
        be used as the probability to select the patrol point as
        target.
        """
        foodList = self.getFoodYouAreDefending(gameState).asList()
        total = 0

        # Get the minimum distance from the food to our
        # patrol points.

        for position in self.noWall:
            closestFoodDistance = 99999
            foodList = self.getFoodYouAreDefending(gameState).asList()
            closestFoodDistance = min([self.getMazeDistance(position, food) for food in foodList])

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
        """
        Select some patrol point to use as target.
        """
        rand = random.random()
        sum = 0.0
        for key in self.defenderList.keys():
            sum += self.defenderList[key]
            if rand < sum:
                return key

    def registerInitialState(self, gameState):
        CaptureAgent.registerInitialState(self, gameState)
        self.distancer.getMazeDistances()

        # Compute central positions without walls from map layout.
        # The defender will walk among these positions to defend
        # its territory.

        self.defenderList = {}
        if self.red:
            middle = (gameState.data.layout.width - 2) / 2
        else:
            middle = ((gameState.data.layout.width - 2) / 2) + 1
        self.noWall = []
        for i in range(1, gameState.data.layout.height - 1):
            if not gameState.hasWall(middle, i):
                self.noWall.append((middle, i))

        while len(self.noWall) > (gameState.data.layout.height - 2) / 2:
            self.noWall.pop(0)
            self.noWall.pop(len(self.noWall) - 1)
        # Update probabilities to each patrol point.
        self.DefendingProbability(gameState)

    def chooseAction(self, gameState):

        # start = time.time()
        #  Patrol probabilities.

        defendingFoodList = self.getFoodYouAreDefending(gameState).asList()
        if self.previousFood and len(self.previousFood) != len(defendingFoodList):
            self.DefendingProbability(gameState)

        CurrentPosition = gameState.getAgentPosition(self.index)
        if CurrentPosition == self.target:
            self.target = None

        # Visible enemy , keep chasing.
        opponentsState = []
        for i in self.getOpponents(gameState):
            opponentsState.append(gameState.getAgentState(i))

        visible = [opponent for opponent in opponentsState if opponent.isPacman and opponent.getPosition() != None]

        if len(visible) > 0:
            positions = [invader.getPosition() for invader in visible]
            minDis, self.target = min(
                [(self.getMazeDistance(CurrentPosition, position), position) for position in positions])

        # If we can't see an invader, but our pacdots were eaten,
        # we will check the position where the pacdot disappeared.
        elif self.previousFood != None:
            eaten = [food for food in self.previousFood if food not in self.getFoodYouAreDefending(gameState).asList()]
            if len(eaten) > 0:
                self.target = eaten.pop()

        # Update the distribution.
        self.previousFood = self.getFoodYouAreDefending(gameState).asList()

        # If we have only a few pacdots, let's walk among them.
        if self.target == None and len(self.getFoodYouAreDefending(gameState).asList()) <= 4:
            food = self.getFoodYouAreDefending(gameState).asList() \
                   + self.getCapsulesYouAreDefending(gameState)
            self.target = random.choice(food)

        # Random patrolling
        elif self.target == None:
            self.target = self.selectPatrolTarget()

        # Choose action. We will take the action that brings us
        # closer to the target. However, we will never stay put
        # and we will never invade the enemy side.

        actions = gameState.getLegalActions(self.index)
        # feasible = []
        # fvalues = []

        feasible = [a for a in actions if not a == Directions.STOP
                    and not gameState.generateSuccessor(self.index, a).getAgentState(self.index).isPacman]
        fvalues = [
            self.getMazeDistance(gameState.generateSuccessor(self.index, a).getAgentPosition(self.index), self.target)
            for a in actions if
            not a == Directions.STOP and not gameState.generateSuccessor(self.index, a).getAgentState(
                self.index).isPacman]


        # Randomly chooses between ties.

        best = min(fvalues)
        ties = filter(lambda x: x[0] == best, zip(fvalues, feasible))

        # print 'eval time for defender agent %d: %.4f' % (self.index, time.time() - start)
        return random.choice(ties)[1]
