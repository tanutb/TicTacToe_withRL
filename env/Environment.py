##### Some part of this code generate from CHATGPT

class TicTacToe:
    def __init__(self):
        self.board = [[" " for _ in range(3)] for _ in range(3)]
        self.current_player = "X"

    def print_board(self):
        for row in self.board:
            print(" | ".join(row))
            print("-" * 9)

    def change_player(self):
        if self.current_player == "X" : self.current_player = "O"
        elif self.current_player == "O" : self.current_player = "X"

    def get_state(self):
        ### change [ ["" , "x" , "0"]] ->
        #### "x" -> 1 ," " -> 0 , "O" -> 2
        state = []
        for row in self.board :
            for value in row : 
                if value == "O": state.append('2')
                elif value == "X": state.append('1')
                elif value == " " : state.append('0')
        return "".join(state)

    def check_winner(self):
        # Check rows
        for row in self.board:
            if row[0] == row[1] == row[2] != " ":
                return row[0]
        # Check columns
        for col in range(3):
            if self.board[0][col] == self.board[1][col] == self.board[2][col] != " ":
                return self.board[0][col]
        # Check diagonals
        if self.board[0][0] == self.board[1][1] == self.board[2][2] != " ":
            return self.board[0][0]
        if self.board[0][2] == self.board[1][1] == self.board[2][0] != " ":
            return self.board[0][2]
        return None

    def is_board_full(self):
        for row in self.board:
            for cell in row:
                if cell == " ":
                    return False
        return True

    def reset(self):
        self.board = [[" " for _ in range(3)] for _ in range(3)]
        self.current_player = "X"
        return self.get_state()

    def step(self , action):
        ### action ( x , y )
        row , col = action
        if self.board[row][col] == " ":
            self.board[row][col] = self.current_player
            winner = self.check_winner()
            if winner:
                reward = 1.0
                return reward , self.get_state() , True
            elif self.is_board_full():
                reward = 0.1
                return reward , self.get_state(), True
            
        return 0.0 , self.get_state(), False


