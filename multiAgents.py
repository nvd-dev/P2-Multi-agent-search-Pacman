# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        foods = newFood.asList()
        ghost_pos = successorGameState.getGhostPositions()
        food_dist = []
        ghost_dist = []

        for food in foods:
            food_dist.append(manhattanDistance(food, newPos))

        for ghost in ghost_pos:
            ghost_dist.append(manhattanDistance(ghost, newPos))

        if currentGameState.getPacmanPosition() == newPos:
            return (-(float("inf")))

        for dist in ghost_dist:
            if dist < 2:
                return (-(float("inf")))

        if len(food_dist) == 0:
            return float("inf")

        return 1000/sum(food_dist) + 10000/len(food_dist)

def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        def min(gameState, agentID, depth):
            actions = gameState.getLegalActions(agentID)
            buffer = None

            if len(actions) == 0:
                return (self.evaluationFunction(gameState), None)
            l = float("inf")

            for action in actions:
                if (agentID == gameState.getNumAgents() - 1):
                    next_value = max(gameState.generateSuccessor(agentID, action), depth + 1)
                else:
                    next_value = min(gameState.generateSuccessor(agentID, action), agentID+1, depth)
                next_value = next_value[0]
                if (next_value < l):
                    l, buffer = next_value, action

            return (l, buffer)

        def max(gameState, depth):
            actions = gameState.getLegalActions(0)
            buffer = None

            if len(actions) == 0 or gameState.isWin() or gameState.isLose() or depth == self.depth:
                return (self.evaluationFunction(gameState), None)
            w = -(float("inf"))

            for action in actions:
                next_value = min(gameState.generateSuccessor(0, action), 1, depth)
                next_value = next_value[0]
                if (next_value > w):
                    w, buffer = next_value, action
            
            return (w, buffer)

        max = max(gameState, 0)[1]
        return max

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"

        def min_value(gameState, agentID, depth, a, b):
            actions = gameState.getLegalActions(
                agentID)  # Get the actions of the ghost
            if len(actions) == 0:
                return (self.evaluationFunction(gameState), None)
            l = float("inf")
            buffer = None
            for action in actions:
                if (agentID == gameState.getNumAgents() - 1):
                    next_value = alpha_beta(gameState.generateSuccessor(
                        agentID, action), depth + 1, a, b)
                else:
                    next_value = min_value(gameState.generateSuccessor(
                        agentID, action), agentID + 1, depth, a, b)
                next_value = next_value[0]
                if (next_value < l):
                    l, buffer = next_value, action

                if (l < a):
                    return (l, buffer)

                b = min(b, l)

            return (l, buffer)

        def alpha_beta(gameState, depth, a, b):
            actions = gameState.getLegalActions(0)
            if len(actions) == 0 or gameState.isWin() or gameState.isLose() or depth == self.depth:
                return (self.evaluationFunction(gameState), None)

            w = -(float("inf"))
            buffer = None
            for action in actions:
                next_value = min_value(
                    gameState.generateSuccessor(0, action), 1, depth, a, b)
                next_value = next_value[0]
                if w < next_value:
                    w, buffer = next_value, action
                if w > b:
                    return (w, buffer)
                a = max(a, w)
            return (w, buffer)

        a = -(float("inf"))
        b = float("inf")
        alpha_beta = alpha_beta(gameState, 0, a, b)[1]
        return alpha_beta

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        def max(gameState, depth):
            actions = gameState.getLegalActions(0)
            if len(actions) == 0 or gameState.isWin() or gameState.isLose() or depth == self.depth:
                return (self.evaluationFunction(gameState), None)

            w = -(float("inf"))
            buffer = None

            for action in actions:
                next_value = expected_val(
                    gameState.generateSuccessor(0, action), 1, depth)
                next_value = next_value[0]
                if (w < next_value):
                    w, buffer = next_value, action
            return (w, buffer)

        def expected_val(gameState, agentID, depth):
            actions = gameState.getLegalActions(agentID)
            if len(actions) == 0:
                return (self.evaluationFunction(gameState), None)

            l = 0
            buffer = None
            for action in actions:
                if (agentID == gameState.getNumAgents() - 1):
                    next_value = max(
                        gameState.generateSuccessor(agentID, action), depth+1)
                else:
                    next_value = expected_val(gameState.generateSuccessor(
                        agentID, action), agentID+1, depth)
                next_value = next_value[0]
                prob = next_value/len(actions)
                l += prob
            return (l, buffer)

        max = max(gameState, 0)[1]
        return max

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    pacman_position = currentGameState.getPacmanPosition()
    ghosts = currentGameState.getGhostStates()
    all_food = currentGameState.getFood()
    caps = currentGameState.getCapsules()

    if currentGameState.isWin():
        return float("inf")

    if currentGameState.isLose():
        return float("-inf")

    food_remain = []
    for food in all_food.asList():
        food_remain += [util.manhattanDistance(food, pacman_position)]

    closest_food = min(food_remain)
    ghost_remaining = []
    all_ghost = []
    for ghost in ghosts:
        if ghost.scaredTimer == 0:
            ghost_remaining += [util.manhattanDistance(
                pacman_position, ghost.getPosition())]
        elif ghost.scaredTimer > 0:
            all_ghost += [util.manhattanDistance(
                pacman_position, ghost.getPosition())]
    closest_ghost = -1

    if len(ghost_remaining) > 0:
        closest_ghost = min(ghost_remaining)
    closest_ghost_reset = -1

    if len(all_ghost) > 0:
        closest_ghost_reset = min(all_ghost)
    score = scoreEvaluationFunction(currentGameState)
    score -= 1.5 * closest_food + 2 * \
        (1.0/closest_ghost) + 2 * closest_ghost_reset + 20 * \
        len(caps) + 4 * len(all_food.asList())
    return score

# Abbreviation
better = betterEvaluationFunction