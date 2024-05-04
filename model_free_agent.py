import matplotlib.pyplot
import torch
import random
import numpy as np
import chess
from chess import engine
from chess import Move
from collections import deque
from model1 import Linear_QNet, QTrainer
import encoder
import random
import matplotlib.pyplot as plt

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

FILES = ['a','b','c','d','e','f','g','h']
RANKS = ['1','2','3','4','5','6','7','8']
PROMORANKS = ['2','7']
PROMOPIECE = ['n','b','r','q']

class ModelFreeAgent:

    def __init__(self):
        self.numGames = 0
        self.epsilon = 0 #parameter to contol randomness
        self.gamma = 0.9 #learning/discount rate
        self.memory = deque(maxlen=MAX_MEMORY) #popleft()
        self.color = None
        self.model = Linear_QNet(64, 2395, 4160)
        self.trainer = QTrainer(self.model, lr = LR, gamma = self.gamma)
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

        array = array.flatten()
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

    def findValidAction(self, game, prediction):

        possibleActions = torch.topk(prediction, 4160).indices
        action = None
        moveFound = False
        count = 0
        move = ""
        while not moveFound and count < 4160:
            action = possibleActions[count].item()
            if action > 4096:
                action = action - 4096
                index3 = action % 4
                index2 = (action // 4) % 2
                index1 = ((action // 4) // 2) % 8

                index2_2 = 0
                if index2 == 0:
                    index2_2 = 1
                else:
                    index2_2 = 8

                move = FILES[index1] + PROMORANKS[index2] + FILES[index1] + str(index2_2) + PROMOPIECE[index3]
            else:
                rank2 = action % 8
                file2 = (action // 8) % 8
                rank1 = ((action // 8) // 8) % 8
                file1 = (((action // 8) // 8) // 8) % 8

                move = FILES[file1] + RANKS[rank1] + FILES[file2] + RANKS[rank2]

            for x in game.legal_moves:
                strX = str(x)
                if move == strX:
                    moveFound = True
                    #print("Move Found")
            if not moveFound:
                count += 1
                
        
        #print("Valid move is ", move, " found at index ", action, " at count ", count)
        actionArray = np.zeros(4160)
        actionArray[action] = 1
        return move, actionArray
    
    def findValidActionRandAction(self, game):
        
        action = None
        moveFound = False
        count = 0
        move = ""
        numbers = list(range(0,4160))
        random.shuffle(numbers)
        while not moveFound and count < 4160:
            action = numbers[count]
            if action > 4096:
                action = action - 4096
                index3 = action % 4
                index2 = (action // 4) % 2
                index1 = ((action // 4) // 2) % 8

                index2_2 = 0
                if index2 == 0:
                    index2_2 = 1
                else:
                    index2_2 = 8

                move = FILES[index1] + PROMORANKS[index2] + FILES[index1] + str(index2_2) + PROMOPIECE[index3]
            else:
                rank2 = action % 8
                file2 = (action // 8) % 8
                rank1 = ((action // 8) // 8) % 8
                file1 = (((action // 8) // 8) // 8) % 8

                move = FILES[file1] + RANKS[rank1] + FILES[file2] + RANKS[rank2]

            for x in game.legal_moves:
                strX = str(x)
                if move == strX:
                    moveFound = True
                    #print("Move Found")
            if not moveFound:
                count += 1
            
        #print("Valid move is ", move, " found at index ", action, " at count ", count)
        actionArray = np.zeros(4160)
        actionArray[action] = 1
        return move, actionArray


    def getAction(self, state, game):
        #random moves: tradeoff exploration / exploitation
        self.epsilon = 80 - self.numGames
        if random.randint(0,200) < self.epsilon:
            move = self.findValidActionRandAction(game)
        else:
            state = state.flatten()
            tensorState = torch.tensor(state, dtype=torch.float)
            prediction = self.model(tensorState)
            move = self.findValidAction(game, prediction) 
        return move

    def quitEngine(self):
        self.temporaryEngine.quit()

    def setColor(self, color):
        self.color = color

    def getReward(self, game, action):
        reward = 0

        parseableAction = str(action)

        if 'x' in parseableAction:
            takenSquare = parseableAction.split('x')[1]
            takenSquare = str(takenSquare)
            if len(takenSquare) == 3:
                takenSquare = takenSquare[:-1]
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

        if game.gives_check((game.parse_uci(action))):
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
    gamesWon = 0
    agent = ModelFreeAgent()
    engine = chess.engine.SimpleEngine.popen_uci(r"D:\repos\CSE-575-Group-Project-AI-Chess\stockfish\stockfish-windows-x86-64-avx2.exe")
    game = chess.Board()

    for x in range(0,500):

        plotScores = []
        num = random.uniform(0,1)
        if num == 0:
            agent.color = chess.WHITE
        else:
            agent.color = chess.BLACK

        gameOver = False
        
        while not gameOver:
            info = engine.analyse(game, chess.engine.Limit(time=0.1))
            if num == 0:
                info = info["score"].white()
            else:
                info = info["score"].black()
            plotScores.append(info)
            if num == 0:
                # get old state
                oldState = agent.getGameState(game)

                # get move
                finalMove, finalMoveIndex = agent.getAction(oldState, game)

                #perform move and get new State
                reward = agent.getReward(game, finalMove)
                game.push(finalMove)
                
                #print(game)

                gameOutcome = game.outcome()
                if gameOutcome != None:
                    gameOver = True
                else:
                    gameOver = False

                newState = agent.getGameState(game)

                #train short memory
                agent.trainShortMemory(oldState, finalMoveIndex, reward, newState, gameOver)

                #remember
                agent.remember(oldState, finalMoveIndex, reward, newState, gameOver)
                
                
                if gameOver:
                    # train long memory, plot result
                    game.reset()
                    agent.numGames += 1
                    agent.trainLongMemory()

                    if gameOutcome.winner == agent.color:
                        gamesWon += 1

                    print('Game: ', agent.numGames)
                    
                    if plotScores:
                        sum = 0
                        countScores = 0
                        for score in plotScores:
                            if '#' in str(score):
                                pass
                            else:
                                sum += int(str(score))
                                countScores += 1
                        average = float(sum) / countScores
                        plotMeanScores.append(average)

                else:
                    result = engine.play(game, chess.engine.Limit(time=0.1))
                    game.push(game.parse_uci(result.move))
                    #print(game)

                    gameOutcome = game.outcome()
                    if gameOutcome != None:
                        gameOver = True
                    else:
                        gameOver = False

                    if gameOver:
                        # train long memory, plot result
                        game.reset()
                        agent.numGames += 1
                        agent.trainLongMemory()

                        if gameOutcome.winner == agent.color:
                            gamesWon += 1

                        print('Game: ', agent.numGames)

                        if plotScores:
                            sum = 0
                            countScores = 0
                            for score in plotScores:
                                if '#' in str(score):
                                    pass
                                else:
                                    sum += int(str(score))
                                    countScores += 1
                            average = float(sum) / countScores
                            plotMeanScores.append(average)
                        
            else:
                result = engine.play(game, chess.engine.Limit(time=0.1))
                game.push(result.move)
                #print(game)

                gameOutcome = game.outcome()
                if gameOutcome != None:
                    gameOver = True
                else:
                    gameOver = False

                if gameOver:
                    # train long memory, plot result
                    game.reset()
                    agent.numGames += 1
                    agent.trainLongMemory()

                    if gameOutcome.winner == agent.color:
                        gamesWon += 1

                    print('Game: ', agent.numGames)

                    if plotScores:
                        sum = 0
                        countScores = 0
                        for score in plotScores:
                            if '#' in str(score):
                                pass
                            else:
                                sum += int(str(score))
                                countScores += 1
                        average = float(sum) / countScores
                        plotMeanScores.append(average)

                else:
                    # get old state
                    oldState = agent.getGameState(game)

                    # get move
                    finalMove, finalMoveIndex = agent.getAction(oldState, game)

                    #perform move and get new State
                    reward = agent.getReward(game, finalMove)
                    game.push(game.parse_uci(finalMove))
                    #print(game)


                    gameOutcome = game.outcome()
                    if gameOutcome != None:
                        gameOver = True
                    else:
                        gameOver = False

                    newState = agent.getGameState(game)

                    #train short memory
                    agent.trainShortMemory(oldState, finalMoveIndex, reward, newState, gameOver)

                    #remember
                    agent.remember(oldState, finalMoveIndex, reward, newState, gameOver)

                    gameOver = game.outcome()

                    if gameOver != None:
                        gameOver = True
                        # train long memory, plot result
                        game.reset()
                        agent.numGames += 1
                        agent.trainLongMemory()

                        if gameOver.winner == agent.color:
                            gamesWon += 1

                        print('Game: ', agent.numGames)

                        if plotScores:
                            sum = 0
                            countScores = 0
                            for score in plotScores:
                                if '#' in str(score):
                                    pass
                                else:
                                    sum += int(str(score))
                                    countScores += 1
                            average = float(sum) / countScores
                            plotMeanScores.append(average)


    gamesList = list(range(agent.numGames))

    fig, ax = plt.subplots()

    ax.plot(gamesList, plotMeanScores)
    ax.set(xlabel='Number of Games', ylabel='Average Evalution of Game', title='Average Evaluation of Game over Number of Games Played')

    ax.grid()
    
    fig.savefig("test.png")

    plt.show()
    
    print("Games won: ", gamesWon)
    engine.quit()
                        


if __name__ == '__main__':
    train()