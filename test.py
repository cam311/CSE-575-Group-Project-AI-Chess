import chess
from chess import engine
import model_free_agent
from model_free_agent import ModelFreeAgent

engine = chess.engine.SimpleEngine.popen_uci(r"D:\repos\CSE-575-Group-Project-AI-Chess\stockfish\stockfish-windows-x86-64-avx2.exe") #change this to your own filepath

board = chess.Board()
agent = ModelFreeAgent()

turn = 0

while not board.is_game_over():
    print("Turn " + str(turn))
    if turn % 2 == 0:
        result = engine.play(board, chess.engine.Limit(time=0.1))
        board.push(result.move)
    else:
        result = agent.getAction(board)
        board.push(result)
    turn += 1
    print(board)

engine.quit()
agent.quitEngine()