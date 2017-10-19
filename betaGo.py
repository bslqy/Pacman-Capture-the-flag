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
from captureAgents import AgentFactory
from game import Directions
import random, time, util
from util import nearestPoint

#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first = 'MasterAAgent', second = 'MasterDAgent'):
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


class EvaluationBasedAgentHelper():
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

  def evaluate(self, gameState, action):
    """
    Computes a linear combination of features and feature weights
    """
    features = self.getFeatures(gameState, action)
    weights = self.getWeights(gameState, action)
    return features * weights

  def getFeatures(self, gameState, action):
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)
    features['successorScore'] = self.cap.getScore(successor)
    return features

  def getWeights(self, gameState, action):
    return {'successorScore': 1.0}


class AttackerAgentHelper(EvaluationBasedAgentHelper):
  "Gera Carlo, o agente ofensivo."

  def __init__(self, index, cap, gameState):
    self.index = index
    self.cap = cap
    # Variables used to verify if the agent os locked
    self.numEnemyFood = "+inf"
    self.inactiveTime = 0
    self.cap.distancer.getMazeDistances()
    self.retreat = False
    if self.cap.red:
      self.midWidth = gameState.data.layout.width / 2 - 5
    else:
      self.midWidth = gameState.data.layout.width / 2 + 5
    self.legalPositions = [p for p in gameState.getWalls().asList(False) if p[1] > 1]
    self.oppoents = self.cap.getOpponents(gameState)
    if self.cap.red:
      centralX = (gameState.data.layout.width - 2) / 2
    else:
      centralX = ((gameState.data.layout.width - 2) / 2) + 1
    self.noWallSpots = []
    for i in range(1, gameState.data.layout.height - 1):
      if not gameState.hasWall(centralX, i):
        self.noWallSpots.append((centralX, i))
        # print self.noWallSpots

  def getFeatures(self, gameState, action):
    """
    Get features used for state evaluation.
    """
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)

    # Compute score from successor state
    features['successorScore'] = self.cap.getScore(successor)

    # Compute remain food
    features['targetFood'] = len(self.cap.getFood(gameState).asList())

    # Compute distance to the nearest food
    foodList = self.cap.getFood(successor).asList()
    if len(foodList) > 0:
      myPos = successor.getAgentState(self.index).getPosition()
      minDistance = min([self.cap.getMazeDistance(myPos, food) for food in foodList])
      features['distanceToFood'] = minDistance

    # Compute the carrying dots
    features['carryDot'] = successor.getAgentState(self.index).numCarrying

    # Compute distance to closest ghost
    myPos = successor.getAgentState(self.index).getPosition()
    enemies = [successor.getAgentState(i) for i in self.cap.getOpponents(successor)]
    inRange = filter(lambda x: not x.isPacman and x.getPosition() != None, enemies)
    if len(inRange) > 0:
      positions = [agent.getPosition() for agent in inRange]
      closest = min(positions, key=lambda x: self.cap.getMazeDistance(myPos, x))
      closestDist = self.cap.getMazeDistance(myPos, closest)
      if closestDist <= 5:
        features['distanceToGhost'] = closestDist

    # Compute if is pacman
    features['isPacman'] = 1 if successor.getAgentState(self.index).isPacman else 0

    # Get the closest distance to the middle of the board.

    features['distanceToMid'] = min([self.cap.distancer.getDistance(myPos, i)
                                     for i in self.noWallSpots])

    # Get whether there is a power pill we are chasing.
    capsulesChasing = None
    if self.cap.red:
        capsulesChasing = gameState.getBlueCapsules()
    else:
        capsulesChasing = gameState.getRedCapsules()

    # distance and minimum distance to the capsule.
    capsulesChasingDistances = [self.cap.distancer.getDistance(myPos, capsule) for capsule in
                                capsulesChasing]
    minCapsuleChasingDistance = min(capsulesChasingDistances) if len(capsulesChasingDistances) else 0

    features['distoCapsule'] = minCapsuleChasingDistance
    return features

  def getWeights(self, gameState, action):
    """
    Get weights for the features used in the evaluation.
    """
    #If tha agent is locked, we will make him try and atack
    if self.inactiveTime > 80:
      return {'successorScore': 10, 'distanceToFood': -10, 'distanceToGhost': 50, 'carryDot': 50,
              'isPacman': 0, 'targetFood': -1000, 'distanceToMid': 0, 'distoCapsule': -500}

    # If opponent is scared, the agent should not care about distanceToGhost
    successor = self.getSuccessor(gameState, action)
    myPos = successor.getAgentState(self.index).getPosition()
    enemies = [successor.getAgentState(i) for i in self.cap.getOpponents(successor)]
    inRange = filter(lambda x: not x.isPacman and x.getPosition() != None, enemies)
#    if len(inRange) > 0:
#      positions = [agent.getPosition() for agent in inRange]
#      closestPos = min(positions, key=lambda x: self.getMazeDistance(myPos, x))
#      closestDist = self.getMazeDistance(myPos, closestPos)
#      closest_enemies = filter(lambda x: x[0] == closestPos, zip(positions, inRange))
#      for agent in closest_enemies:
#        if agent[1].scaredTimer > 3:
#          return {'successorScore': 2, 'distanceToFood': -500, 'distanceToGhost': 0,'carryDot': 50,
#                  'isPacman': 0, 'targetFood': -1000, 'distanceToMid': 0,'distoCapsule':0}

    # Weights normally used

    scaredTimes = gameState.getAgentState(self.oppoents[0]).scaredTimer
    if scaredTimes > 3:
        return {'successorScore': 2, 'distanceToFood': -500, 'distanceToGhost': 0,'carryDot': 100,
                  'isPacman': 0, 'targetFood': -100, 'distanceToMid': -10,'distoCapsule':0}
    elif self.retreat:
        return {'successorScore': 10, 'distanceToFood': 0, 'distanceToGhost': 500, 'carryDot': 50,
                'isPacman': -100, 'targetFood': 20, 'distanceToMid': -100,'distoCapsule':0}
    else:
        return {'successorScore': 10, 'distanceToFood': -500, 'distanceToGhost': 50, 'carryDot': 50,
                'isPacman': 0,'targetFood': -1000, 'distanceToMid': 0,'distoCapsule':-500}

  def randomSimulation(self, depth, gameState):
    """
    Random simulate some actions for the agent. The actions other agents can take
    are ignored, or, in other words, we consider their actions is always STOP.
    The final state from the simulation is evaluated.
    """
    new_state = gameState.deepCopy()
    while depth > 0:
      # Get valid actions
      actions = new_state.getLegalActions(self.index)
      # The agent should not stay put in the simulation
      actions.remove(Directions.STOP)
      current_direction = new_state.getAgentState(self.index).configuration.direction
      # The agent should not use the reverse direction during simulation
      reversed_direction = Directions.REVERSE[new_state.getAgentState(self.index).configuration.direction]
      if reversed_direction in actions and len(actions) > 1:
        actions.remove(reversed_direction)
      # Randomly chooses a valid action
      a = random.choice(actions)
      # Compute new state and update depth
      new_state = new_state.generateSuccessor(self.index, a)
      depth -= 1
    # Evaluate the final simulation state
    return self.evaluate(new_state, Directions.STOP)

  def takeToEmptyAlley(self, gameState, action, depth):
    """
    Verify if an action takes the agent to an alley with
    no pacdots.
    """
    if depth == 0:
        return False
    # if self.retreat:
    #     return True
    # else:
    targetFood = len(self.cap.getFood(gameState).asList())
    new_state = gameState.generateSuccessor(self.index, action)
    new_targetFood = len(self.cap.getFood(new_state).asList())
    if new_targetFood < targetFood:
        return False
    actions = new_state.getLegalActions(self.index)
    actions.remove(Directions.STOP)
    reversed_direction = Directions.REVERSE[new_state.getAgentState(self.index).configuration.direction]
    if reversed_direction in actions:
      actions.remove(reversed_direction)
    if len(actions) == 0:
      return True
    for a in actions:
      if not self.takeToEmptyAlley(new_state, a, depth - 1):
        return False
    return True





  # Implemente este metodo para controlar o agente (1s max).
  def chooseAction(self, gameState):
    # You can profile your evaluation time by uncommenting these lines
    # start = time.time()
    # If some of our food was eaten, we need to update
    #scaredTimes = gameState.getAgentState(self.enemies[0]).scaredTimer
    if self.cap.getScore(gameState) < 4:
        carryLimit = 3
    else:
        carryLimit = 2

    if gameState.getAgentState(self.index).numCarrying < carryLimit and len(self.cap.getFood(gameState).asList()) > 2:
        self.retreat = False
    else:
        self.retreat = True

    # Updates inactiveTime. This variable indicates if the agent is locked.
    currentEnemyFood = len(self.cap.getFood(gameState).asList())
    if self.numEnemyFood != currentEnemyFood:
      self.numEnemyFood = currentEnemyFood
      self.inactiveTime = 0
    else:
      self.inactiveTime += 1
    # If the agent dies, inactiveTime is reseted.
    if gameState.getInitialAgentPosition(self.index) == gameState.getAgentState(self.index).getPosition():
      self.inactiveTime = 0

    # Get valid actions. Staying put is almost never a good choice, so
    # the agent will ignore this action.
    all_actions = gameState.getLegalActions(self.index)
    all_actions.remove(Directions.STOP)
    actions = []
    for a in all_actions:
      if not self.takeToEmptyAlley(gameState, a, 8):
        actions.append(a)
    if len(actions) == 0:
      actions = all_actions
    reversed_direction = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
    if reversed_direction in actions and len(actions) >= 2:
      actions.remove(reversed_direction)
    fvalues = []
    for a in actions:
      new_state = gameState.generateSuccessor(self.index, a)
      value = 0
      for i in range(1, 31):
        value += self.randomSimulation(10, new_state)
      fvalues.append(value)

    best = max(fvalues)
    ties = filter(lambda x: x[0] == best, zip(fvalues, actions))
    toPlay = random.choice(ties)[1]

    # print 'eval time for offensive agent %d: %.4f' % (self.index, time.time() - start)
    return toPlay


class DefenderAgentHelper():
  "Gera Monte, o agente defensivo."
  def __init__(self, index,cap,gameState):
    #CaptureAgent.__init__(self, index)
    self.index = index
    self.cap = cap
    self.target = None
    self.lastObservedFood = None
    # This variable will store our patrol points and
    # the agent probability to select a point as target.
    self.patrolDict = {}
    if self.cap.red:
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

  def distFoodToPatrol(self, gameState):
    """
    This method calculates the minimum distance from our patrol
    points to our pacdots. The inverse of this distance will
    be used as the probability to select the patrol point as
    target.
    """
    food = self.cap.getFoodYouAreDefending(gameState).asList()
    total = 0

    # Get the minimum distance from the food to our
    # patrol points.
    for position in self.noWallSpots:
      closestFoodDist = "+inf"
      for foodPos in food:
        dist = self.cap.getMazeDistance(position, foodPos)
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


  # Implemente este metodo para controlar o agente (1s max).
  def chooseAction(self, gameState):
    # You can profile your evaluation time by uncommenting these lines
    # start = time.time()
    # our patrol points probabilities.

    if self.lastObservedFood and len(self.lastObservedFood) != len(self.cap.getFoodYouAreDefending(gameState).asList()):
      self.distFoodToPatrol(gameState)

    mypos = gameState.getAgentPosition(self.index)
    if mypos == self.target:
      self.target = None

    # If we can see an invader, we go after him.
    x = self.cap.getOpponents(gameState)
    enemies = [gameState.getAgentState(i) for i in self.cap.getOpponents(gameState)]
    invaders = filter(lambda x: x.isPacman and x.getPosition() != None, enemies)
    if len(invaders) > 0:
      positions = [agent.getPosition() for agent in invaders]
      self.target = min(positions, key=lambda x: self.cap.getMazeDistance(mypos, x))
    # If we can't see an invader, but our pacdots were eaten,
    # we will check the position where the pacdot disappeared.
    elif self.lastObservedFood != None:
      eaten = set(self.lastObservedFood) - set(self.cap.getFoodYouAreDefending(gameState).asList())
      if len(eaten) > 0:
        self.target = eaten.pop()
    # Update the agent memory about our pacdots.
    self.lastObservedFood = self.cap.getFoodYouAreDefending(gameState).asList()

    # No enemy in sight, and our pacdots are not disappearing.
    # If we have only a few pacdots, let's walk among them.
    if self.target == None and len(self.cap.getFoodYouAreDefending(gameState).asList()) <= 4:
      food = self.cap.getFoodYouAreDefending(gameState).asList() \
             + self.cap.getCapsulesYouAreDefending(gameState)
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
      if not a == Directions.STOP:
        newpos = new_state.getAgentPosition(self.index)
        goodActions.append(a)
        fvalues.append(self.cap.getMazeDistance(newpos, self.target))

    # Randomly chooses between ties.
    best = min(fvalues)
    ties = filter(lambda x: x[0] == best, zip(fvalues, goodActions))

    # print 'eval time for defender agent %d: %.4f' % (self.index, time.time() - start)
    return random.choice(ties)[1]

class MasterDAgent(CaptureAgent):
  def __init__(self, index):
    CaptureAgent.__init__(self, index)

  def registerInitialState(self, gameState):
    CaptureAgent.registerInitialState(self, gameState)
    self.defA = DefenderAgentHelper(self.index, self, gameState)
    self.attA = AttackerAgentHelper(self.index, self, gameState)

  def chooseAction(self, gameState):
    self.enemies = self.getOpponents(gameState)
    invaders = [a for a in self.enemies if gameState.getAgentState(a).isPacman]
    numInvaders = len(invaders)

    # Check if we have the poison active.
    scaredTimes = [gameState.getAgentState(enemy).scaredTimer for enemy in self.enemies]
    if numInvaders == 0 and self.getScore(gameState) < 10:
      return self.attA.chooseAction(gameState)
    else:
      print(self.defA.target,"Target is ...................")
      return self.defA.chooseAction(gameState)


class MasterAAgent(CaptureAgent):
  def __init__(self, index):
    CaptureAgent.__init__(self, index)

  def registerInitialState(self, gameState):
    CaptureAgent.registerInitialState(self, gameState)
    self.defA = DefenderAgentHelper(self.index, self, gameState)
    self.attA = AttackerAgentHelper(self.index, self, gameState)

  def chooseAction(self, gameState):
    self.enemies = self.getOpponents(gameState)
    invaders = [a for a in self.enemies if gameState.getAgentState(a).isPacman]
    numInvaders = len(invaders)

    # Check if we have the poison active.
    scaredTimes = [gameState.getAgentState(enemy).scaredTimer for enemy in self.enemies]
    if numInvaders == 2 or self.getScore(gameState) > 9:
      return self.defA.chooseAction(gameState)
    else:
      return self.attA.chooseAction(gameState)
