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
    features['successorScore'] = self.cap.getScore(successor) - self.cap.getScore(gameState)
    return features

  def getWeights(self, gameState, action):
    return {'successorScore': 1.0}


class AttackerAgentHelper(EvaluationBasedAgentHelper):
  "Gera Carlo, o agente ofensivo."


  def getFeatures(self, gameState, action):
    """
    Get features used for state evaluation.
    """
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)

    # Compute score from successor state
    features['successorScore'] = self.cap.getScore(successor) - self.cap.getScore(gameState)

    # Compute remain food
    features['targetFood'] = len(self.cap.getFood(gameState).asList())

    # Compute distance to the nearest food
    foodList = self.cap.getFood(successor).asList()
    if len(foodList) > 0:
      myPos = successor.getAgentState(self.index).getPosition()
      minDistance = min([self.cap.getMazeDistance(myPos,food) for food in foodList])
      features['distanceToFood'] = minDistance

    # Compute the carrying dots
    features['carryDot'] = successor.getAgentState(self.index).numCarrying

    # Compute distance to closest ghost
    myPos = successor.getAgentState(self.index).getPosition()
    enemies = [successor.getAgentState(i) for i in self.cap.getOpponents(successor)]
    inRange = filter(lambda x: not x.isPacman and x.getPosition() != None, enemies)
    if len(inRange) > 0:
      positions = [agent.getPosition() for agent in inRange]
      closest = min(positions, key=lambda x: self.cap.getMazeDistance(myPos,x))
      closestDist = self.cap.getMazeDistance(myPos,closest) - 1
      if closestDist <= 5:
        features['distanceToGhost'] = closestDist
      if closestDist == 1:
        features['carryDot'] = 0


    # Get the closest distance to the middle of the board.

    features['distanceToMid'] = min([self.cap.distancer.getDistance(myPos, i)
                                     for i in self.noWallSpots])
    features['homeSide'] = 0 if successor.getAgentState(self.index).isPacman else 1
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
    
      # If opponent is scared, the agent should not care about GhostDistance
      successor = self.getSuccessor(gameState, action)
      opponents = [successor.getAgentState(i) for i in self.cap.getOpponents(successor)]
      visible = filter(lambda x: not x.isPacman and x.getPosition() != None, opponents)
      if len(visible) > 0:
          for agent in visible:
              if agent.scaredTimer > 0:
                  if agent.scaredTimer > 10:
                      return {'successorScore': 50, 'distanceToFood': -20, 'distanceToGhost': 0,
                              'distanceToCapsule': 0, 'distanceToMid': 0, 'carryDot': 20}
                  elif 4 < agent.scaredTimer < 10:
                      return {'successorScore': 110, 'distanceToFood': -10, 'distanceToGhost': -5,
                              'distanceToCapsule': -10, 'distanceToMid': -10,'carryDot': 20 }
            
              # Visible and not scared
              else:
                  return {'successorScore': 110, 'distanceToFood': -10, 'distanceToGhost': 20,
                          'distanceToCapsule': -15, 'distanceToMid': -10,'carryDot': 0}
                
                
      # Attacker only try to defence if it is close to it (less than 4 steps)
      enemiesPacMan = [successor.getAgentState(i) for i in self.cap.getOpponents(successor)]
      Range = filter(lambda x: x.isPacman and x.getPosition() != None, enemiesPacMan)
      if len(Range) > 0 and not successor.getAgentState(self.index).isPacman:
          return {'successorScore': 0, 'distanceToFood': -3, 'distanceToCapsule': 0,
                  'distanceToGhost': 0,'distanceToMid': 0, 'carryDot': 0}
    
      # Did not see anything
      return {'successorScore': 30, 'distanceToFood': -8, 'distanceToGhost': 0,
              'distanceToCapsule': -6, 'distanceToMid': 0, 'carryDot': 35}

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

  def __init__(self, index,cap,gameState):
    self.index = index
    self.cap = cap
    # Variables used to verify if the agent os locked
    self.numEnemyFood = "+inf"
    self.inactiveTime = 0
    self.cap.distancer.getMazeDistances()
    self.retreat = False
    if self.cap.red:
		self.midWidth = gameState.data.layout.width / 2 - 1
    else:
		self.midWidth = gameState.data.layout.width / 2 + 1
    self.legalPositions = [p for p in gameState.getWalls().asList(False) if p[1] > 1]
    self.oppoents = self.cap.getOpponents(gameState)
    if self.cap.red:
      centralX = gameState.data.layout.width/ 2 - 1
    else:
      centralX = gameState.data.layout.width/2 + 1
    self.noWallSpots = []
    for i in range(1, gameState.data.layout.height - 1):
      if not gameState.hasWall(centralX, i):
        self.noWallSpots.append((centralX, i))
    #print self.noWallSpots





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
      for i in range(1, 40):
        value += self.randomSimulation(10, new_state)
      value = value/31
      fvalues.append(value)

    best = max(fvalues)
    ties = filter(lambda x: x[0] == best, zip(fvalues, actions))
    toPlay = random.choice(ties)[1]
    if self.retreat:
        for a in actions:
            feature = self.getFeatures(gameState,a)
            if feature['homeSide'] == 1:
                return a

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
        _,dist = mazeDistance(position,foodPos,gameState)
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
      self.target = min(positions, key=lambda x: self.cap.getMazeDistance(mypos,x))
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
        fvalues.append(self.cap.getMazeDistance(self.target,newpos))

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

            self.enemies = self.getOpponents(gameState)
            invaders = [a for a in self.enemies if gameState.getAgentState(a).isPacman]
            myPos = gameState.getAgentState(self.index).getPosition()
            numInvaders = len(invaders)
            foods = self.getFood(gameState).asList()
            capsules = self.getCapsules(gameState)
            for e in capsules:
                foods.append(e)


            scaredTimes = min([gameState.getAgentState(enemy).scaredTimer for enemy in self.enemies])
            dis = 1000#mazeDistance(enemyPos,myPos,gameState)
            for enemy in self.enemies:
                enemyPos = gameState.getAgentPosition(enemy)

                if enemyPos and (not gameState.getAgentState(enemy).isPacman) and gameState.getAgentState(enemy).scaredTimer == 0 :
                    print enemyPos
                    print  "detected enermy, using monte carlo"
                    print "*********************"
                    _,temp = mazeDistance(myPos,enemyPos,gameState)
                    if temp < dis:
                        dis = temp
            if dis < 7:
                return self.attA.chooseAction(gameState)

            distanceToHome = 1000
            actionToHome = gameState.getLegalActions(self.index)
            mid = gameState.data.layout.width/2 - 1
            if not self.red:
                mid += 1
            legalPositions = [p for p in gameState.getWalls().asList(False) if p[1] > 1]
            boader = [p for p in legalPositions if p[0] == mid]

            for e in boader:
                tempA,tempD = self.eat(e,gameState)
                if tempD < distanceToHome and tempD != 0:
                    distanceToHome = tempD
                    actionToHome = tempA[0]

            if scaredTimes > 0:
                if len(capsules) > 0:
                    foods = [e for e in foods if e not in capsules]
                    distanceTocapsules = 1000
                    actionTocapsules = gameState.getLegalActions(self.index)
                    for e in capsules:
                        tempA,tempD = self.eat(e,gameState)
                        if tempD < distanceTocapsules:
                            distanceTocapsules = tempD
                            actionTocapsules = tempA[0]
                        if distanceTocapsules >= scaredTimes+2 :
                            print "chi da li wan!!!!!"
                            print "**********************************"
                            return actionTocapsules
                else:

                    if distanceToHome >= scaredTimes  or gameState.getAgentState(self.index).numCarrying > 5:
                        print "back to my home"
                        print "********************************************"
                        return actionToHome
                # road to home
            print "take food"
            print  "************************************"
            if len(foods) > 0:
                actionToFood,distanceToFood = self.eat(foods[0],gameState)
                actionToFood = actionToFood[0]
                for e in foods:
                    tempA,tempD = self.eat(e,gameState)
                    if tempD < distanceToFood:
                        distanceToFood = tempD
                        actionToFood = tempA[0]
            if (gameState.getAgentState(self.index).numCarrying > 5 and scaredTimes == 0) or len(foods) == 0:
                return actionToHome
            return actionToFood
        else:
            return self.defA.chooseAction(gameState)

    def eat(self,position,gameState):
      start = gameState.getAgentPosition(self.index)
      actions,length = mazeDistance(start,position,gameState,True)
      return actions,length

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
        myPos = gameState.getAgentState(self.index).getPosition()
        numInvaders = len(invaders)
        foods = self.getFood(gameState).asList()
        capsules = self.getCapsules(gameState)
        for e in capsules:
            foods.append(e)


        scaredTimes = min([gameState.getAgentState(enemy).scaredTimer for enemy in self.enemies])
        dis = 1000#mazeDistance(enemyPos,myPos,gameState)
        for enemy in self.enemies:
            enemyPos = gameState.getAgentPosition(enemy)

            if enemyPos and (not gameState.getAgentState(enemy).isPacman) and gameState.getAgentState(enemy).scaredTimer == 0 :

                print  "detected enermy, using monte carlo"
                print "*********************"
                _,temp = mazeDistance(myPos,enemyPos,gameState)
                if temp < dis:
                    dis = temp
        if dis < 7:
            return self.attA.chooseAction(gameState)

        distanceToHome = 1000
        actionToHome = gameState.getLegalActions(self.index)
        mid = gameState.data.layout.width/2 -1
        if not self.red:
            mid += 1
        legalPositions = [p for p in gameState.getWalls().asList(False) if p[1] > 1]
        boader = [p for p in legalPositions if p[0] == mid]

        for e in boader:
            tempA,tempD = self.eat(e,gameState)
            if tempD < distanceToHome and tempD != 0:
                distanceToHome = tempD
                actionToHome = tempA[0]

        if scaredTimes > 0:
            if len(capsules) > 0:
                foods = [e for e in foods if e not in capsules]
                distanceTocapsules = 1000
                actionTocapsules = gameState.getLegalActions(self.index)
                for e in capsules:
                    tempA,tempD = self.eat(e,gameState)
                    if tempD < distanceTocapsules:
                        distanceTocapsules = tempD
                        actionTocapsules = tempA[0]
                if distanceTocapsules == scaredTimes:
                    print "chi da li wan!!!!!"
                    print "**********************************"
                    return actionTocapsules
            else:

                if distanceToHome >= scaredTimes :
                    print "back to my home"
                    print "********************************************"
                    return actionToHome
                # road to home
        print "take food"
        print  "************************************"
        if len(foods) > 0:
                actionToFood,distanceToFood = self.eat(foods[0],gameState)
                actionToFood = actionToFood[0]
                for e in foods:
                    tempA,tempD = self.eat(e,gameState)
                    if tempD < distanceToFood:
                        distanceToFood = tempD
                        actionToFood = tempA[0]
        if (gameState.getAgentState(self.index).numCarrying > 5 and scaredTimes == 0) or len(foods) == 0:
                return actionToHome
        return actionToFood


    def eat(self,position,gameState):
      start = gameState.getAgentPosition(self.index)
      actions,length = mazeDistance(start,position,gameState,True)
      return actions,length





import heapq

class PriorityQueue:

    def  __init__(self):
        self.heap = []
        self.count = 0

    def push(self, item, priority):
        entry = (priority, self.count, item)
        heapq.heappush(self.heap, entry)
        self.count += 1

    def pop(self):
        (_, _, item) = heapq.heappop(self.heap)
        return item

    def isEmpty(self):
        return len(self.heap) == 0

    def update(self, item, priority):
        # If item already in priority queue with higher priority, update its priority and rebuild the heap.
        # If item already in priority queue with equal or lower priority, do nothing.
        # If item not in priority queue, do the same thing as self.push.
        for index, (p, c, i) in enumerate(self.heap):
            if i == item:
                if p <= priority:
                    break
                del self.heap[index]
                self.heap.append((priority, c, item))
                heapq.heapify(self.heap)
                break
        else:
            self.push(item, priority)

from game import Actions

def mazeDistance(point1, point2, gameState,returnA = False):
    """
    Returns the maze distance between any two points, using the search functions
    you have already built. The gameState can be any game state -- Pacman's
    position in that state is ignored.

    Example usage: mazeDistance( (2,4), (5,6), gameState)

    This might be a useful helper function for your ApproximateSearchAgent.
    """
    x1, y1 = point1
    x2, y2 = point2
    x1,y1 = int(x1),int(y1)
    x2,y2 = int(x2),int(y2)
    walls = gameState.getWalls()
    assert not walls[x1][y1], 'point1 is a wall: ' + str(point1)
    assert not walls[x2][y2], 'point2 is a wall: ' + str(point2)
    prob = PositionSearchProblem(gameState, start=point1, goal=point2, warn=False, visualize=False)
    list = bfs(prob)

    return list,len(list)


class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


class PositionSearchProblem(SearchProblem):
    """
    A search problem defines the state space, start state, goal test, successor
    function and cost function.  This search problem can be used to find paths
    to a particular point on the pacman board.

    The state space consists of (x,y) positions in a pacman game.

    Note: this search problem is fully specified; you should NOT change it.
    """

    def __init__(self, gameState, costFn = lambda x: 1, goal=(1,1), start=None, warn=True, visualize=True):
        """
        Stores the start and goal.

        gameState: A GameState object (pacman.py)
        costFn: A function from a search state (tuple) to a non-negative number
        goal: A position in the gameState
        """
        self.walls = gameState.getWalls()
        self.startState = start
        if start != None: self.startState = start
        self.goal = goal
        self.costFn = costFn
        self.visualize = visualize
        if warn and (gameState.getNumFood() != 1 or not gameState.hasFood(*goal)):
            print 'Warning: this does not look like a regular search maze'

        # For display purposes
        self._visited, self._visitedlist, self._expanded = {}, [], 0 # DO NOT CHANGE

    def getStartState(self):
        return self.startState

    def isGoalState(self, state):
        isGoal = state == self.goal

        # For display purposes only
        if isGoal and self.visualize:
            self._visitedlist.append(state)
            import __main__
            if '_display' in dir(__main__):
                if 'drawExpandedCells' in dir(__main__._display): #@UndefinedVariable
                    __main__._display.drawExpandedCells(self._visitedlist) #@UndefinedVariable

        return isGoal

    def getSuccessors(self, state):
        """
        Returns successor states, the actions they require, and a cost of 1.

         As noted in search.py:
             For a given state, this should return a list of triples,
         (successor, action, stepCost), where 'successor' is a
         successor to the current state, 'action' is the action
         required to get there, and 'stepCost' is the incremental
         cost of expanding to that successor
        """

        successors = []
        for action in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
            x,y = state
            dx, dy = Actions.directionToVector(action)
            nextx, nexty = int(x + dx), int(y + dy)
            if not self.walls[nextx][nexty]:
                nextState = (nextx, nexty)
                cost = self.costFn(nextState)
                successors.append( ( nextState, action, cost) )

        # Bookkeeping for display purposes
        self._expanded += 1 # DO NOT CHANGE
        if state not in self._visited:
            self._visited[state] = True
            self._visitedlist.append(state)

        return successors

    def getCostOfActions(self, actions):
        """
        Returns the cost of a particular sequence of actions. If those actions
        include an illegal move, return 999999.
        """
        if actions == None: return 999999
        x,y= self.getStartState()
        cost = 0
        for action in actions:
            # Check figure out the next state and see whether its' legal
            dx, dy = Actions.directionToVector(action)
            x, y = int(x + dx), int(y + dy)
            if self.walls[x][y]: return 999999
            cost += self.costFn((x,y))
        return cost

def breadthFirstSearch(problem):
		from game import Directions
		queue = util.Queue()
		startPoint = problem.getStartState()
		track = []
		visited = set([])
		dict = {'South':Directions.SOUTH,'North':Directions.NORTH,'West':Directions.WEST,'East':Directions.EAST}
		map = {}
		queue.push(startPoint)
		while not queue.isEmpty():
			point = queue.pop()
			visited.add(point)
			if problem.isGoalState(point):
				break
			else:
				successors = problem.getSuccessors(point)
				for e in successors:
					if (e[0] not in visited) and (e[0] not in queue.list):
						queue.push(e[0])
						map[e[0]] = (point,e[1])

		def keyName(key):
			if key in map.keys():
				track.append(map[key][1])
				keyName(map[key][0])
		keyName(point)
		return [x for x in list(reversed(track))]



bfs = breadthFirstSearch
