import chess
from chess import engine

engine = chess.engine.SimpleEngine.popen_uci(r"D:\repos\CSE-575-Group-Project-AI-Chess\stockfish\stockfish-windows-x86-64-avx2.exe") #change this to your own filepath

board = chess.Board()
while not board.is_game_over():
    result = engine.play(board, chess.engine.Limit(time=0.1))
    board.push(result.move)
    print("Turn\n")

engine.quit()