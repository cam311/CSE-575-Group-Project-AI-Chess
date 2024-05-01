import chess
from chess import engine
import model_free_agent
from model_free_agent import ModelFreeAgent
import numpy as np

engine = chess.engine.SimpleEngine.popen_uci(r"D:\repos\CSE-575-Group-Project-AI-Chess\stockfish\stockfish-windows-x86-64-avx2.exe") #change this to your own filepath

board = chess.Board()
#agent = ModelFreeAgent()

stringBoard = str(board)

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
            
            
        count += 1
print(array)

print(board.legal_moves)

turn = 0

while board.is_game_over():
    print("Turn " + str(turn))
    if turn % 2 == 0:
        result = engine.play(board, chess.engine.Limit(time=0.1))
        move = result.move
        parseableMove = board.san(move)
        board.push(result.move)
    else:
        result = engine.play(board, chess.engine.Limit(time=0.1))
        move = result.move
        parseableMove = board.san(move)
        board.push(result.move)
        #result = agent.getAction(board)
        #board.push(result)
    turn += 1
    if 'x' in parseableMove:
        takenSquare = parseableMove.split('x')[1]
        takenPiece = board.piece_at(chess.parse_square(takenSquare))
        match takenPiece.piece_type:
            case chess.PAWN:
                print("Takes 1 point of material")
            case chess.KNIGHT:
                print("Takes 3 points of material")
            case chess.BISHOP:
                print("Takes 3 points of material")
            case chess.ROOK:
                print("Takes 5 points of material")
            case chess.QUEEN:
                print("Takes 9 points of material")
    

engine.quit()
#agent.quitEngine()