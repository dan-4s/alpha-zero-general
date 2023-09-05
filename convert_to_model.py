"""
Utility for converting checkpoints to model files.

Written by Danilo Vucetic, github: dan-4s
"""
import glob
from othello.pytorch.OthelloNNet import OthelloNNet
from othello.OthelloGame import OthelloGame
from tictactoe.pytorch.TicTacToeNNet import TicTacToeNNet
from tictactoe.TicTacToeGame import TicTacToeGame
import torch
from utils import dotdict

args = dotdict({
    'lr': 0.001,
    'dropout': 0.3,
    'epochs': 10,
    'batch_size': 64,
    'cuda': torch.cuda.is_available(),
    'num_channels': 512,
})
if __name__ == "__main__":
    for file in glob.glob("temp/alphazero_othello_*.pth"):
        check = torch.load(file)
        game = OthelloGame(n=6)
        nnet = OthelloNNet(game, args)
        # game = TicTacToeGame()
        # nnet = TicTacToeNNet(game, args)
        nnet.load_state_dict(check['state_dict'])
        del check['state_dict']
        check['model'] = nnet
        torch.save(obj=check, f=f'model_outputs/{file[5:-1]}')


