import streamlit as st
from RL_Agent import DoubleQLearning, QLearning, SARSA, DeepQLearning
import pandas as pd

@st.cache_resource
class TicTacToe:
    def __init__(self):
        self.board = [[" " for _ in range(3)] for _ in range(3)]
        self.current_player = "X"

    def print_board(self):
        custom_css = """
        <style>
        table {
        text-align: center; /* Center align text */
        width: 300px; /* Fixed width for the table */
        border-collapse: collapse; /* Collapse table borders */
        }
        tbody {
        text-align: center; /* Center align text */
        }

        td, th {
        padding: 10px; /* Padding inside cells */
        text-align: center; /* Center align text */
        border: 1px solid black; /* Border for cells */
        width: 100px; /* Fixed width for cells */
        height: 50px; /* Fixed height for cells */
        }

        td {text-align: center;}
        </style>
        """

        # Display the custom CSS using st.markdown
        st.markdown(custom_css, unsafe_allow_html=True)

        # Display the DataFrame using st.table
        st.table(self.board)


    def change_player(self):
        self.current_player = "O" if self.current_player == "X" else "X"

    def get_state(self):
        state = []
        for row in self.board:
            for value in row:
                if value == "O":
                    state.append('2')
                elif value == "X":
                    state.append('1')
                else:
                    state.append('0')
        return "".join(state)

    def check_winner(self):
        for row in self.board:
            if row[0] == row[1] == row[2] != " ":
                return row[0]
        for col in range(3):
            if self.board[0][col] == self.board[1][col] == self.board[2][col] != " ":
                return self.board[0][col]
        if self.board[0][0] == self.board[1][1] == self.board[2][2] != " ":
            return self.board[0][0]
        if self.board[0][2] == self.board[1][1] == self.board[2][0] != " ":
            return self.board[0][2]
        return None

    def is_board_full(self):
        return all(cell != " " for row in self.board for cell in row)

    def reset(self):
        self.board = [[" " for _ in range(3)] for _ in range(3)]
        self.current_player = "X"
        return self.get_state()

    def step(self, action):
        row, col = action
        if self.board[row][col] == " ":
            self.board[row][col] = self.current_player
            winner = self.check_winner()
            if winner:
                reward = 100.0
                return reward, self.get_state(), True
            elif self.is_board_full():
                reward = 50
                return reward, self.get_state(), True
            else:
                reward = -5
                return reward, self.get_state(), False
        return 0.0, self.get_state(), False
    
@st.cache_resource
class model:
    def __init__(self) -> None:
        self.QLearning_p1, self.QLearning_p2 = QLearning.QLearningAgent(), QLearning.QLearningAgent()
        self.QLearning_p1.load('QLearning_p1')
        self.QLearning_p2.load('QLearning_p2')
    

class Deploy:
    def __init__(self) -> None:
        self.env = TicTacToe()
        self.model = model()

    def play_game(self):
        if st.button("Clear"):
            self.env.reset()
        st.title('Tic Tac Toe with Reinforcement Learning')
        st.session_state.selected_player = st.radio("Select 1st or 2nd player:", ('X', 'O'))
        if st.session_state.selected_player == self.env.current_player:
            action = st.number_input("Enter index (0-8):", min_value=0, max_value=8, step=1, key=f"player_move_{self.env.current_player}")
            if st.button("Make Move"):
                row_index = action // 3
                col_index = action % 3
                player_action = (row_index, col_index)
                reward, next_state, done = self.env.step(player_action)
                # st.write(f"Player ({self.env.current_player}) put in {player_action}")

                if done:
                    st.write(f"Player WIN")


                self.env.change_player()
                state = next_state
                action = self.model.QLearning_p2.get_max_action(state)

                _, next_state, done = self.env.step(action)
                # st.write(f"BOT ({self.env.current_player}) put in {action}")
                self.env.change_player()
                state = next_state

                if done:
                    st.write(f"BOT WIN :)")

        else : 
            action = st.number_input("Enter index (0-8):", min_value=0, max_value=8, step=1, key=f"player_move_{self.env.current_player}")
            bot_action = self.model.QLearning_p1.get_max_action(self.env.get_state())
            reward, next_state, done = self.env.step(bot_action)
            # st.write(f"BOT ({self.env.current_player}) put in {action}")
            if done:
                st.write(f"BOT WIN :)")

            self.env.change_player()

            state = next_state
            if st.button("Make Move"):

                row_index = action // 3
                col_index = action % 3
                player_action = (row_index, col_index)

                _, next_state, done = self.env.step(player_action)
                # st.write(f"PLAYER ({self.env.current_player}) put in {player_action}")
                self.env.change_player()
                state = next_state

                if done:
                    st.write(f"PLAYER WIN ")
                    
        self.env.print_board()


if __name__ == "__main__":
    deploy = Deploy()
    deploy.play_game()