import torch
import random
import numpy as np
import chess
from chess import engine
from collections import deque
from model1 import Linear_QNet, QTrainer

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

class ModelFreeAgent:

    def __init__(self):
        self.numGames = 0
        self.epsilon = 0 #parameter to contol randomness
        self.gamma = 0 #learning/discount rate
        self.memory = deque(maxlen=MAX_MEMORY) #popleft()
        self.color = None
        self.model = None
        self.trainer = None
        #self.temporaryEngine = chess.engine.SimpleEngine.popen_uci(r"D:\repos\CSE-575-Group-Project-AI-Chess\stockfish\stockfish-windows-x86-64-avx2.exe")

    def getGameState(self, game):
        stringBoard = str(game)

        array = np.zeros((8,8))
        splitBoard = stringBoard.split()
        count = 0
        for i in range(0,8):
            for j in range(0,8):
                match splitBoard[count]:
                    case 'k':
                        array[i][j] = -6
                    case 'q':
                        array[i][j] = -5
                    case 'r':
                        array[i][j] = -4    
                    case 'n':
                        array[i][j] = -3
                    case 'b':
                        array[i][j] = -2
                    case 'p':
                        array[i][j] = -1
                    case '.':
                        array[i][j] = 0
                    case 'P':
                        array[i][j] = 1
                    case 'B':
                        array[i][j] = 2
                    case 'N':
                        array[i][j] = 3
                    case 'R':
                        array[i][j] = 4
                    case 'Q':
                        array[i][j] = 5
                    case 'K':
                        array[i][j] = 6
        return array

    def remember(self, state, action, reward, nextState, gameOver):
        self.memory.append((state, action, reward, nextState, gameOver)) # popleft if MAX_MEMORY is reached

    def trainLongMemory(self):
        if len(self.memory) > BATCH_SIZE:
            miniSample = random.sample(self.memory, BATCH_SIZE) # list of tuples
        else:
            miniSample = self.memory

        states, actions, rewards, nextStates, gameOvers = zip(*miniSample)
        self.trainer.trainStep(states, actions, rewards, nextStates, gameOvers)

    def trainShortMemory(self, state, action, reward, nextState, gameOver):
        self.trainer.trainStep(state, action, reward, nextState, gameOver)
        
    def getAction(self, state):
        #random moves: tradeoff exploration / exploitation
        self.epsilon = 80 - self.numGames
        if random.randint(0,200) < self.epsilon:

            moveList = list(state.legal_moves)
            moveIndex = random.randint(0, state.legal_moves.count() - 1)
            move = moveList[moveIndex]
        else:
            tensorState = torch.tensor(state, dtype=torch.float)
            prediction = self.model.predict(tensorState)
            move = torch.argmax(prediction).item()
            #execute move, TODO 

        return move

    def quitEngine(self):
        self.temporaryEngine.quit()

    def setColor(self, color):
        self.color = color

    def getReward(self, game, action):
        reward = 0

        parseableAction = game.san(action)

        if 'x' in parseableAction:
            takenSquare = parseableAction.split('x')[1]
            takenPiece = game.piece_at(chess.parse_square(takenSquare))
            match takenPiece.piece_type:
                case chess.PAWN:
                    reward += 1
                case chess.KNIGHT:
                    reward += 3
                case chess.BISHOP:
                    reward += 3
                case chess.ROOK:
                    reward += 5
                case chess.QUEEN:
                    reward += 9

        if game.gives_check(action):
            reward += 20
        outcome = game.outcome()
        if outcome != None:
            if outcome.winner == self.color:
                reward += 100

        return reward
def train():
    plotScores = []
    plotMeanScores = []
    totalScore = 0
    record = 0
    agent = ModelFreeAgent()
    game = chess.Board()

    while True:
        # get old state
        oldState = agent.getGameState(game)

        # get move
        finalMove = agent.getAction(oldState)

        #perform move and get new State
        game.push(finalMove)

        reward = agent.getReward(finalMove)
        gameOver = game.outcome()
        if gameOver != None:
            gameOver = True

        newState = agent.getGameState(game)

        #train short memory
        agent.trainShortMemory(oldState, finalMove, reward, newState, gameOver)

        #remember
        agent.remember(oldState, finalMove, reward, newState, gameOver)

        if gameOver:
            # train long memory, plot result
            game.reset()
            agent.numGames += 1
            agent.trainLongMemory()

            print('Game ', agent.numGames)

            #todo plot



if __name__ == '__main__':
    train()