import torch
import random
import numpy as np
import chess
from chess import engine
from collections import deque

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

class ModelFreeAgent:

    def __init__(self):
        self.numGames = 0
        self.epsilon = 0 #parameter to contol randomness
        self.gamma = 0 #learning/discount rate
        self.memory = deque(maxlen=MAX_MEMORY) #popleft()
        self.temporaryEngine = chess.engine.SimpleEngine.popen_uci(r"D:\repos\CSE-575-Group-Project-AI-Chess\stockfish\stockfish-windows-x86-64-avx2.exe")

    def getGameState(self, game):
        return game.board()

    def remember(self, state, action, reward, nextState, gameOver):
        pass

    def trainLongMemory(self):
        pass

    def trainShortMemory(self, state, action, reward, nextState, gameOver):
        pass
        
    def getAction(self, state):
        self.epsilon = 80 - self.numGames
        if random.randint(0,200) < self.epsilon:

            moveList = list(state.legal_moves)
            moveIndex = random.randint(0, state.legal_moves.count() - 1)
            move = moveList[moveIndex]
        else:
            result = self.temporaryEngine.play(state, chess.engine.Limit(time=0.1))
            move = result.move

        return move

    def quitEngine(self):
        self.temporaryEngine.quit()

def train():
    pass

if __name__ == '__main__':
    pass