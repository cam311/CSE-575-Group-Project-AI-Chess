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

    def getGameState(self, game):
        pass

    def remember(self, state, action, reward, nextState, gameOver):
        pass

    def trainLongMemory(self):
        pass

    def trainShortMemory(self, state, acstion, reward, next_state, done):
        pass

    def getAction(self, state):
        pass

def train():
    plotScores = []
    plotMeanScores = []
    totalScore = 0
    record = 0
    agent = ModelFreeAgent()
    board = chess.Board()

    while True:
        #get old state
        stateOld = agent.getGameState(game)

        #get move
        finalMove = agent.getAction(state)

        #perform move and get new state
        reward, done, score = game.play_step(final_move)
        stateNew = agent.get_state(game)

        #train short memory
        agent.trainShortMemory(state_old, final_move, reward, state_new, done)

        #remember
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            #train long memory, plot result
            
            board.reset()

            agent.numGames += 1
            agent.trainLongMemory()
            
            if score > record:
                record = score
                agent.model.save()

            print('Game', agent.n_games, 'Score', score, 'Record:', record)




if __name__ == '__main__':
    train()