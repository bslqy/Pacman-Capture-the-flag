# myTeam.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to
# http://inst.eecs.berkeley.edu/~cs188/pacman/pacman.html
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).

from captureAgents import CaptureAgent, AgentFactory
import random, util, operator
from util import nearestPoint
from game import Directions

POWERCAPSULETIME = 40


#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first='UpAgent', second='DownAgent'):
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


class AvatarFactory(AgentFactory):
    def __init__(self, isRed):
        AgentFactory.__init__(self, isRed)
        self.agents = ['upAgent', 'downAgent']

    def getAgent(self, index):
        if len(self.agents) > 0:
            agent = self.agents.pop()
            if agent == 'upAgent':
                return UpAgent(index)
        return DownAgent(index)

##########
# Agents #
##########

class ReflexCaptureAgent(CaptureAgent):
    # Give each agent a most likely position and a power timer
    def __init__(self, gameState):
        CaptureAgent.__init__(self, gameState)
        self.mostlikely = [None] * 4
        self.powerTimer = 0

    def registerInitialState(self, gameState):

        CaptureAgent.registerInitialState(self, gameState)

        # Sets if agent is on red team or not
        if self.red:
            CaptureAgent.registerTeam(self, gameState.getRedTeamIndices())
        else:
            CaptureAgent.registerTeam(self, gameState.getBlueTeamIndices())

        # Get how large the game space is
        # Legal positions are positions without walls
        height = gameState.data.layout.height
        width = gameState.data.layout.width
        self.legalPositions = []
        for y in range(1, height):
            for x in range(1, width):
                if not gameState.hasWall(x, y):
                    self.legalPositions.append((x, y))

        global beliefs
        beliefs = [util.Counter()] * gameState.getNumAgents()

        # All beliefs begin with the agent at its inital position
        for i, val in enumerate(beliefs):
            if i in self.getOpponents(gameState):
                beliefs[i][gameState.getInitialAgentPosition(i)] = 1.0

                # Agents inital move towards the centre with a bias for either the top or the bottom
        self.goToCenter(gameState)

    # (Done)Detect position of enemies that are visible
    def getEnemies(self, gameState):
        opponents = self.getOpponents(gameState)
        enemies = []
        for o in opponents:
            i = gameState.getAgentState(o).getPosition()
            if i != None:
                enemies.append((o, i))
        return enemies

    # (Done)Find which enemy is the closest
    def getDistToEnemy(self, gameState):
        enemies = self.getEnemies(gameState)
        myPos = gameState.getAgentPosition(self.index)
        minDist = None
        if len(enemies) > 0:
            minDist = float('inf')
            for i, e in enemies:
                dist = self.getMazeDistance(myPos, e)
                if minDist > dist:
                    minDist = dist
        return minDist

    # (Done)Calculates the distance to the partner of the current agent
    def getDistToAlly(self, gameState):
        if self.index == self.agentsOnTeam[0]:
            dist = None
        else:
        # only the pacman cares the distance between the ally
            allyIndex = self.agentsOnTeam[0]
            myPos = gameState.getAgentPosition(self.index)
            allyPos = gameState.getAgentState(allyIndex).getPosition()
            dist = self.getMazeDistance(myPos, allyPos)
            if dist == 0:
                dist = 1
        return dist

    # (Done)Which side of the board is the agent?
    def side(self, gameState):
        width = gameState.data.layout.width
        mypos = gameState.getAgentPosition(self.index)
        if self.index % 2 == 1 and mypos[0] < width / 2:
                return 1.0
        elif self.index % 2 == 0 and mypos[0] > width / 2:
                return 1.0
        else:
                return 0.0

    # Gets the distribution for where a ghost could be, all weight equally
    def getWeightofAction(self, gameState, pos):
        posActions = [(pos[0] - 1, pos[1]), (pos[0] + 1, pos[1]), (pos[0], pos[1] - 1),
                      (pos[0], pos[1] + 1), (pos[0], pos[1])]
        actions = []
        for act in posActions:
            if act in self.legalPositions:
                actions.append(act)

        weight = util.Counter()
        for a in actions:
            weight[a] = 1
        return weight

    # Looks at how an agent could move from where they currently are
    def elapseTime(self, gameState):
        opponents = self.getOpponents(gameState)
        for o, belief in enumerate(beliefs):
            if o in opponents:
                newBeliefs = util.Counter()
                # Checks to see what we can actually see
                pos = gameState.getAgentPosition(o)
                if pos != None:
                    newBeliefs[pos] = 1.0
                else:
                    # Look at all current beliefs
                    for p in belief:
                        if p in self.legalPositions and belief[p] > 0:
                            # Check that all these values are legal positions
                            newPosDist = self.getWeightofAction(gameState, p)
                            for x, y in newPosDist:  # iterate over these probabilities
                                newBeliefs[x, y] += belief[p] * newPosDist[x, y]
                                # The new chance is old chance * prob of this location from p
                    if len(newBeliefs) == 0:
                        oldState = self.getPreviousObservation()
                        if oldState != None and oldState.getAgentPosition(o) != None:  # just ate an enemy
                            newBeliefs[oldState.getInitialAgentPosition(o)] = 1.0
                        else:
                            for p in self.legalPositions: newBeliefs[p] = 1.0
                beliefs[o] = newBeliefs

    # Looks for where the enemies currently are
    def observe(self, agent, noisyDistance, gameState):
        myPos = gameState.getAgentPosition(self.index)
        # Current state probabilities
        allPossible = util.Counter()
        for p in self.legalPositions:  # check each legal position
            trueDistance = util.manhattanDistance(p, myPos)  # distance between this point and Pacman
            allPossible[p] += gameState.getDistanceProb(trueDistance, noisyDistance)
            # The new values are product of prior probability and new probability
        for p in self.legalPositions:
            beliefs[agent][p] *= allPossible[p]

    def chooseAction(self, gameState):
        """
        Picks among the actions with the highest Q(s,a).
        """

        opponents = self.getOpponents(gameState)
        # Get noisey distance data
        noisyDistance = gameState.getAgentDistances()
        # Get this agent's current position

        # Observe each opponent to get noisey distance measurement and process
        for o in opponents:
            self.observe(o, noisyDistance[o], gameState)

        # Normalise new probabilities and pick most likely location for enemy agent
        for o in opponents:
            beliefs[o].normalize()
            self.mostlikely[o] = max(beliefs[o].iteritems(), key=operator.itemgetter(1))[0]

        # Do next time step
        self.elapseTime(gameState)
        # Get agent position
        agentPos = gameState.getAgentPosition(self.index)

        ##################
        # Choose Tactics #
        ##################


        # Default to attack mode
        mode = 'offend'

        # Start in the start state, move to the centre then switch to attack
        if self.atCenter == False:
            mode = 'start'

        # If at centre, switch to attack
        if agentPos == self.center and self.atCenter == False:
            self.atCenter = True
            mode = 'offend'

        # If an enemy is attacking our food, hunt that enemy down
        for o in opponents:
            if (gameState.getAgentState(o).isPacman):
                mode = 'patrol'

        # If we directly see an enemy on our side, swich to defence
        enemies = self.getEnemies(gameState)
        if len(enemies) > 0:
            for i, e in enemies:
                if self.getMazeDistance(agentPos, e) < 5 and \
                        not gameState.getAgentState(self.index).isPacman:
                    mode = 'defend'
                    break

        # actions to choose if use the Monte Carlo Tree
        # actions = gameState.getLegalActions(self.index)
        # score = []
        # for a in actions:
        #    state = gameState.generateSuccessor(self.index, a)
        #    s = 0
        #    for i in range(1, 10):
        #        simulationState = self.simulate(5, state)
        #        s += self.evaluate(simulationState, Directions.STOP, evaluateType)
        #        score.append((s, a))
        #
        # highestValue = float('-Inf')
        # bestActions = []
        # for s in score:
        #    if highestValue < s[0]:
        #        highestValue = s[0]
        #        bestActions.append(s[1])
        #
        # return random.choice(bestActions)

        actions = gameState.getLegalActions(self.index)
        # Calcualte heuristic score of each action
        values = [self.evaluate(gameState, a, mode) for a in actions]

        maxValue = max(values)
        bestActions = [a for a, v in zip(actions, values) if v == maxValue]

        return random.choice(bestActions)

    # Random Simulatio of Monte Carlo Tree
    # def simulate(self, depth, gameState):
    #     simulationState = gameState.deepCopy()
    #     while depth > 0:
    #         actions = simulationState.getLegalActions(self.index)
    #         actions.remove(Directions.STOP)
    #         reverse = Directions.REVERSE[simulationState.getAgentState(self.index).configuration.direction]
    #         if reverse in actions and len(actions) > 1:
    #             actions.remove(reverse)
    #         a = random.choice(actions)
    #         simulationState = simulationState.generateSuccessor(self.index, a)
    #         depth -= 1
    #     return simulationState

    # (Done)
    def getSuccessor(self, gameState, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """
        successor = gameState.generateSuccessor(self.index, action)
        pos = successor.getAgentState(self.index).getPosition()
        if pos != nearestPoint(pos):
            return successor.generateSuccessor(self.index, action)
        else:
            return successor

    # (Done) Calculate the heurisic score of each action depending on what tactic is being used
    def evaluate(self, gameState, action, mode):
        """
        Computes a linear combination of features and feature weights
        """
        if mode == 'offend':
            features = self.getOffendFeatures(gameState, action)
            weights = self.getOffendWeights(gameState, action)
        elif mode == 'defend':
            features = self.getDefendFeatures(gameState, action)
            weights = self.getDefendWeights(gameState, action)
        elif mode == 'start':
            features = self.getStartFeatures(gameState, action)
            weights = self.getStartWeights(gameState, action)
        elif mode == 'patrol':
            features = self.getPatrolFeatures(gameState, action)
            weights = self.getPatrolWeights(gameState, action)

        return features * weights


    # Returns all the heuristic features for the ATTACK tactic
    def getOffendFeatures(self, gameState, action):
        features = util.Counter()
        successor = self.getSuccessor(gameState, action)
        # Get own position, size of game state and locations of all food to eat

        myPos = successor.getAgentState(self.index).getPosition()

        width = gameState.data.layout.width
        height = gameState.data.layout.height
        foods = self.getFood(successor).asList()

        # Get score for successor state
        features['successorScore'] = self.getScore(successor)

        # Dist to nearest food heuristic
        if len(foods) > 0:
            minDist = min([self.getMazeDistance(myPos, f) for f in foods])
            features['distanceToFood'] = minDist

        # Pickup food heuristic
        if len(foods) > 0:
            features['pickupFood'] = -len(foods) + 100 * self.getScore(successor)

        # Compute distance to enemy
        distToEnemy = self.getDistToEnemy(successor)
        if (distToEnemy != None):
            if (distToEnemy <= 2):
                features['danger'] = 4 / distToEnemy
            elif (distToEnemy <= 4):
                features['danger'] = 1
            else:
                features['danger'] = 0

        # Compute distance to capsule
        capsules = self.getCapsules(successor)
        if (len(capsules) > 0):
            minDist = min([self.getMazeDistance(myPos, c) for c in capsules])
            features['pickupCapsule'] = -len(capsules)
        else:
            minDist = .1
        features['distanceToCapsule'] = 1.0 / minDist

        # Holding food heuristic
        if myPos in self.getFood(gameState).asList():
            self.foodNum += 1.0
        if self.side(gameState) == 0.0:
            self.foodNum = 0.0
            features['holdFood'] = 0
        else:
            entry = []
            for i in range(1, height):
                if not gameState.hasWall(width / 2, i):
                    entry.append((width / 2, i))
            dist = min(self.distancer.getDistance(myPos, e) for e in entry)
            features['holdFood'] = self.foodNum * dist

        # Dropping off food heuristic
        features['dropFood'] = self.foodNum * (self.side(gameState))

        # If picked up a capsule, set power timer
        if myPos in self.getCapsules(gameState):
            self.powerTimer = POWERCAPSULETIME

        # If powered, reduce power timer each iteration
        if self.powerTimer > 0:
            self.powerTimer -= 1

        # Is powered heuristic
        if(self.powerTimer > 0):
            features['isPowered'] = self.powerTimer / POWERCAPSULETIME
            features['holdFood'] = 0.0
            features['pickupFood'] = 100 * features['pickupFood']
        else:
            features['isPowered'] = 0.0

        # Compute distance to partner
        # if successor.getAgentState(self.index).isPacman:
        #     distanceToAlly = self.getDistToAlly(successor)
        #     if distanceToAlly != None:
        #         features['distanceToAlly'] = 1.0 / distanceToAlly

        # Dead end heuristic
        actions = gameState.getLegalActions(self.index)
        if (len(actions) <= 2):
            features['blindAlley'] = 1.0
        else:
            features['blindAlley'] = 0.0

        # Stop heuristic
        if (action == Directions.STOP):
            features['stop'] = 1.0
        else:
            features['stop'] = 0.0

        return features

    # Returns all the heuristic features for the DEFEND tactic
    def getDefendFeatures(self, gameState, action):
        features = util.Counter()
        successor = self.getSuccessor(gameState, action)
        myPos = successor.getAgentState(self.index).getPosition()

        # List invaders we can see
        enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
        invaders = [e for e in enemies if e.isPacman and e.getPosition() != None]

        # Get number of invaders
        features['numOfInvaders'] = len(invaders)
        if len(invaders) > 0:
            enemyDist = [self.getMazeDistance(myPos, enemy.getPosition()) for enemy in invaders]
            # Find closest invader
            features['distanceToInvaders'] = min(enemyDist)

        # Compute distance to enemy
        distToEnemy = self.getDistToEnemy(successor)
        if (distToEnemy <= 5):
            features['danger'] = 1
            if (distToEnemy <= 1 and successor.getAgentState(self.index).scaredTimer > 0):
                features['danger'] = -1
        else:
            features['danger'] = 0

        # Compute distance to partner
        if successor.getAgentState(self.index).isPacman:
            distanceToAlly = self.getDistToAlly(successor)
            if distanceToAlly != None:
                features['distanceToAlly'] = 1.0 / distanceToAlly

        if action == Directions.STOP:
            features['stop'] = 1
        reverse = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
        if action == reverse:
            features['reverse'] = 1

        return features

    # Returns all the heuristic features for the START tactic
    def getStartFeatures(self, gameState, action):
        features = util.Counter()
        successor = self.getSuccessor(gameState, action)
        myPos = successor.getAgentState(self.index).getPosition()

        # Compute distance to board centre
        dist = self.getMazeDistance(myPos, self.center)
        features['distanceToCenter'] = dist
        if myPos == self.center:
            features['atCenter'] = 1
        return features

    # Returns all the heuristic features for the HUNT tactic
    def getPatrolFeatures(self, gameState, action):

        features = util.Counter()
        successor = self.getSuccessor(gameState, action)
        myPos = successor.getAgentState(self.index).getPosition()

        # Get opponents and invaders
        opponents = self.getOpponents(gameState)
        invaders = [o for o in opponents if successor.getAgentState(o).isPacman]

        # Find number of invaders
        features['numOfInvaders'] = len(invaders)

        # For each invader, calulate its most likely poisiton and distance
        for i in invaders:
            enemyPos = self.mostlikely[i]
            enemyDist = self.getMazeDistance(myPos, enemyPos)
            features['distanceToInvaders'] = enemyDist

        # Compute distance to partner
        if successor.getAgentState(self.index).isPacman:
            distanceToAlly = self.getDistToAlly(successor)
            # distanceToAgent is always None for one of the agents (so they don't get stuck)
            if distanceToAlly != None:
                features['distanceToAlly'] = 1.0 / distanceToAlly

        if action == Directions.STOP:
            features['stop'] = 1
        reverse = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
        if action == reverse:
            features['reverse'] = 1

        return features

    # Returns heuristic weightings for the ATTACK tactic
    def getOffendWeights(self, gameState, action):
        return {'successorScore': 800, 'distanceToFood': -10, 'danger': -1000,
                'pickupFood': 4000, 'distanceToCapsule': 700, 'stop': -1000, 'blindAlley': -200,
                'isPowered': 5000000, 'dropFood': 100, 'holdFood': -20,
                'distanceToAlly': -6000, 'pickupCapsule': 5000}

    # Returns heuristic weightings for the HUNT tactic
    def getPatrolWeights(self, gameState, action):

        return {'numOfInvaders': -100, 'distanceToInvaders': -10, 'stop': -5000,
                'reverse': -5000, 'distanceToAlly': -2500}

        # Returns heuristic weightings for the DEFEND tactic

    def getDefendWeights(self, gameState, action):
        return {'numOfInvaders': -10000, 'distanceToInvaders': -500, 'stop': -5000,
                'reverse': -200, 'danger': 3000, 'distanceToAlly': -4000}

    # Returns heuristic weightings for the START tactic
    def getStartWeights(self, gameState, action):
        return {'distanceToCenter': -1, 'atCenter': 1000}


# (Done)Agent that has a bias to moving around the top of the board
class UpAgent(ReflexCaptureAgent):
    def goToCenter(self, gameState):
        self.entry = []
        self.atCenter = False

        if self.red:
            entryPos = (gameState.data.layout.width / 2 - 1)
        else:
            entryPos = (gameState.data.layout.width / 2)
        height = gameState.data.layout.height
        self.center = (entryPos, height/2)
        for e in range(height/2, height-1):
            if not gameState.hasWall(entryPos, e):
                self.entry.append((entryPos, e))

        myPos = gameState.getAgentState(self.index).getPosition()
        minDist = float('inf')
        minPos = None

        # Find shortest distance to centre
        for e in self.entry:
            dist = self.getMazeDistance(myPos, e)
            if minDist >= dist:
                minDist = dist
                minPos = e

        self.center = minPos


# (Done)Agent that has a bias to moving around the bottom of the board
class DownAgent(ReflexCaptureAgent):
    def goToCenter(self, gameState):
        self.entry = []
        self.atCenter = False
        if self.red:
            entryPos = (gameState.data.layout.width / 2 - 1)
        else:
            entryPos = (gameState.data.layout.width / 2)

        height = gameState.data.layout.height
        self.center = (entryPos, height / 2)
        for e in range(1, height / 2):
            if not gameState.hasWall(entryPos, e):
                self.entry.append((entryPos, e))

        myPos = gameState.getAgentState(self.index).getPosition()
        minDist = float('inf')
        minPos = None

        # Find shortest distance to centre
        for e in self.entry:
            dist = self.getMazeDistance(myPos, e)
            if minDist >= dist:
                minPos = e
                minDist = dist

        self.center = minPos
