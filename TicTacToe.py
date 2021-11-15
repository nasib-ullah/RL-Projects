# Developed by Nasibullah during RL project course in M.Tech
# Algorithm: Q-Learning


import random
from copy import deepcopy
#import csv
#import matplotlib.pyplot as plt
#%matplotlib inline

# States as integer : manual coding
EMPTY = 0
PLAYER_X = 1
PLAYER_O = 2
DRAW = 3

BOARD_FORMAT = """----------------------------
| {0} | {1} | {2} |
|--------------------------|
| {3} | {4} | {5} |
|--------------------------|
| {6} | {7} | {8} |
----------------------------"""
NAMES = [' ', 'X', 'O']


def printboard(state):
    """ Print the board from the internal state."""
    cells = []
    for i in range(3):
        for j in range(3):
            cells.append(NAMES[state[i][j]].center(6))
    print(BOARD_FORMAT.format(*cells))


def emptystate():
    """ An empty 3x3 state."""
    return [[EMPTY, EMPTY, EMPTY], [EMPTY, EMPTY, EMPTY], [EMPTY, EMPTY, EMPTY]]


def gameover(state):
    """ Check if the state is gameover or not."""
    for i in range(3):
        if state[i][0] != EMPTY and state[i][0] == state[i][1] and state[i][0] == state[i][2]:
            return state[i][0]
        if state[0][i] != EMPTY and state[0][i] == state[1][i] and state[0][i] == state[2][i]:
            return state[0][i]
    if state[0][0] != EMPTY and state[0][0] == state[1][1] and state[0][0] == state[2][2]:
        return state[0][0]
    if state[0][2] != EMPTY and state[0][2] == state[1][1] and state[0][2] == state[2][0]:
        return state[0][2]
    for i in range(3):
        for j in range(3):
            if state[i][j] == EMPTY:
                return EMPTY
    return DRAW


def last_to_act(state):
    """ Count who should play."""
    countx = 0
    counto = 0
    for i in range(3):
        for j in range(3):
            if state[i][j] == PLAYER_X:
                countx += 1
            elif state[i][j] == PLAYER_O:
                counto += 1
    if countx == counto:
        return PLAYER_O
    if countx == (counto + 1):
        return PLAYER_X
    return -1


def enumstates(state, idx, agent):
    """ Enumerate the different states from a state."""
    if idx > 8:
        player = last_to_act(state)
        if player == agent.player:
            agent.add(state)
    else:
        winner = gameover(state)
        if winner != EMPTY:
            return
        i = idx // 3
        j = idx % 3
        for val in range(3):
            state[i][j] = val
            enumstates(state, idx + 1, agent)


class Agent(object):
    """ A RL agent abstraction."""

    def __init__(self, player, verbose=False, lossval=0, learning=True):
        """ Create a RL agent."""
        self.values = {}
        self.player = player
        self.verbose = verbose
        self.lossval = lossval
        self.learning = learning
        self.epsilon = 0.01
        self.alpha = 0.99
        self.prevstate = None
        self.prevscore = 0
        self.count = 0
        enumstates(emptystate(), 0, self)

    def episode_over(self, winner):
        """ Backup and reset self.prevstate and self.prevscore."""
        self.backup(self.winnerval(winner))
        self.prevstate = None
        self.prevscore = 0

    def action(self, state):
        """ Play an action (epsilon-drunk policy between random and greedy)."""
        r = random.random()
        if r < self.epsilon:
            move = self.random(state)
            self.log('>>>>>>> Exploratory action: ' + str(move))
        else:
            move = self.greedy(state)
            self.log('>>>>>>> Best action: ' + str(move))
        state[move[0]][move[1]] = self.player
        self.prevstate = self.statetuple(state)
        self.prevscore = self.lookup(state)
        state[move[0]][move[1]] = EMPTY
        return move

    def random(self, state):
        """ Random policy !"""
        available = []
        for i in range(3):
            for j in range(3):
                if state[i][j] == EMPTY:
                    available.append((i, j))
        return random.choice(available)

    def greedy(self, state):
        """ Naive implementation of the greedy policy."""
        maxval = -50000
        maxmove = None
        if self.verbose:
            cells = []
        for i in range(3):
            for j in range(3):
                if state[i][j] == EMPTY:
                    state[i][j] = self.player
                    val = self.lookup(state)
                    state[i][j] = EMPTY
                    if val > maxval:
                        maxval = val
                        maxmove = (i, j)
                    if self.verbose:
                        cells.append('{0:.3f}'.format(val).center(6))
                elif self.verbose:
                    cells.append(NAMES[state[i][j]].center(6))
        if self.verbose:
            print(BOARD_FORMAT.format(*cells))
        self.backup(maxval)
        return maxmove

    def backup(self, nextval):
        """ Backup the next value."""
        if self.prevstate is not None and self.learning:
            self.values[self.prevstate] += self.alpha * (nextval - self.prevscore)

    def lookup(self, state):
        """ Lookup a state."""
        key = self.statetuple(state)
        if key not in self.values:
            self.add(key)
        return self.values[key]

    def add(self, state):
        """ Add a state."""
        winner = gameover(state)
        tup = self.statetuple(state)
        self.values[tup] = self.winnerval(winner)

    def winnerval(self, winner):
        """ Return the value of the winner (0, .5, 1, or self.lossval)."""
        if winner == self.player:
            return 1
        elif winner == EMPTY:
            return 0.5
        elif winner == DRAW:
            return 0
        else:
            return self.lossval

    def printvalues(self):
        """ Print the current internal values."""
        vals = deepcopy(self.values)
        for key in vals:
            state = [list(key[0]), list(key[1]), list(key[2])]
            cells = []
            for i in range(3):
                for j in range(3):
                    if state[i][j] == EMPTY:
                        state[i][j] = self.player
                        cells.append(str(self.lookup(state)).center(3))
                        state[i][j] = EMPTY
                    else:
                        cells.append(NAMES[state[i][j]].center(3))
            print(BOARD_FORMAT.format(*cells))

    def statetuple(self, state):
        """ Return a tuple of tuple for the current state."""
        return (tuple(state[0]), tuple(state[1]), tuple(state[2]))

    def log(self, s):
        """ Print if verbose."""
        if self.verbose:
            print(s)


class Human(object):
    """ An interactive player. """
    def __init__(self, player):
        """ Create an interactive player."""
        self.player = player

    def action(self, state):
        """ Ask (with input(...)) the user to play."""
        printboard(state)
        action = str(input('Your move? '))
        return (int(action.split(',')[0]), int(action.split(',')[1]))

    def episode_over(self, winner):
        """ Check if you win."""
        if winner == DRAW:
            print('Game over! It was a draw.')
        else:
            print('Game over! Winner: Player {0}'.format(winner))


def play(agent1, agent2):
    """ Play once."""
    state = emptystate()
    for i in range(9):
        if i % 2 == 0:
            move = agent1.action(state)
        else:
            move = agent2.action(state)
        state[move[0]][move[1]] = (i % 2) + 1
        winner = gameover(state)
        if winner != EMPTY:
            return winner
    return winner


def measure_performance_vs_random(agent1, agent2):
    """ A naive way to measure performance of two agents vs random."""
    epsilon1 = agent1.epsilon
    epsilon2 = agent2.epsilon
    agent1.epsilon = 0
    agent2.epsilon = 0
    agent1.learning = False
    agent2.learning = False
    r1 = Agent(1)
    r2 = Agent(2)
    r1.epsilon = 1
    r2.epsilon = 1
    probs = [0, 0, 0, 0, 0, 0]
    games = 100
    for i in range(games):
        winner = play(agent1, r2)
        if winner == PLAYER_X:
            probs[0] += 1.0 / games
        elif winner == PLAYER_O:
            probs[1] += 1.0 / games
        else:
            probs[2] += 1.0 / games
    for i in range(games):
        winner = play(r1, agent2)
        if winner == PLAYER_O:
            probs[3] += 1.0 / games
        elif winner == PLAYER_X:
            probs[4] += 1.0 / games
        else:
            probs[5] += 1.0 / games
    agent1.epsilon = epsilon1
    agent2.epsilon = epsilon2
    agent1.learning = True
    agent2.learning = True
    return probs


#---------------------Self Play -----------------------#

p1 = Agent(1, lossval=-1)
p2 = Agent(2, lossval=-1)
r1 = Agent(1, learning=False)
r2 = Agent(2, learning=False)
r1.epsilon = 1
r2.epsilon = 1
print('Training Agents by self playing 100000 games. Please Wait.')
for i in range(100000):
    if i % 20000 == 0:
        print('Self Played : {0} games'.format(i))
        probs = measure_performance_vs_random(p1, p2)
        #print('Performance:{0}'.format(probs))
    winner = play(p1, p2)
    p1.episode_over(winner)
    winner = play(r1, p2)
    p2.episode_over(winner)

print('Agent has trained. Lets play a game now')

#----------------------- Play Game --------------------------#

print(' [ You: X and AI: O    Input guide: for top left block 0,0 and 2,2 for bottom right corner ]')
while True:
	
        p2.verbose = False
        p1 = Human(1)
        winner = play(p1, p2)
        p1.episode_over(winner)
        p2.episode_over(winner)



