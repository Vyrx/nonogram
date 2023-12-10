import random
import copy
from collections import namedtuple
import numpy as np

CheckInfo = namedtuple('CheckInfo', ['matched', 'mismatched', 'violated'])

class Nonogram(object):
    def __init__(self, row_count:int, col_count: int, fill_rate: float):
        """
        Args:
            row: Number of rows for the board.
            col: Number of columns for the board.
            fill_rate: Rate of filled cells.
        """
        
        self.row_count = row_count
        self.col_count = col_count
        self.total_count = self.row_count * self.col_count
        self.fill_rate = fill_rate
        
        self.board = [ [0 for _ in range(col_count)] for _ in range(row_count)]
        self.board_answer = copy.deepcopy(self.board)

        # Initialize board layout
        samples = random.sample(range(self.total_count), int(self.total_count * fill_rate))
        for sample in samples:
            self.board_answer[sample // self.col_count][sample % self.col_count] = 1

        self.row_clues = [] * self.row_count
        self.col_clues = [] * self.col_count
        
        for r in range(row_count):
            row = self.__get_row(self.board_answer, r)
            self.row_clues.append(self.__encode_list(row))
        
        for c in range(col_count):
            col = self.__get_col(self.board_answer, c)
            self.col_clues.append(self.__encode_list(col))

        # Count total number of col clues
        self.col_clues_count = sum([len(item) for item in self.col_clues])
    def __repr__(self) -> str:
        output = "board:\n"
        
        for i, row in enumerate(self.board):
            for cell in row:
                output = output + ('#' if cell else '.') + " "
            output += "| "
            for clue in self.row_clues[i]: 
                output += str(clue) + " "
            output = output + "\n"
        for i in range(self.col_count): 
            output += "- "
        output += "\n"
        for j in range(max([len(item) for item in self.col_clues])):
            for i in range(self.col_count):
                if len(self.col_clues[i]) > j: 
                    output += str(self.col_clues[i][j])
                else:
                    output += " "
                output += " "
            output += "\n"
        return output

    def show_board(self) -> None:
        """Print out the game board."""
        print(self)

    def show_answer(self) -> None:
        output = "answer:\n"
        for i, row in enumerate(self.board_answer):
            for cell in row:
                output = output + ('#' if cell else '.') + " "
            output += "| "
            for clue in self.row_clues[i]: 
                output += str(clue) + " "
            output = output + "\n"
        for i in range(self.col_count): 
            output += "- "
        output += "\n"
        for j in range(max([len(item) for item in self.col_clues])):
            for i in range(self.col_count):
                if len(self.col_clues[i]) > j: 
                    output += str(self.col_clues[i][j])
                else:
                    output += " "
                output += " "
            output += "\n"
        print(output)

    def check(self) -> CheckInfo:
        """
        Check the board with clues and compare it with answer board.
        """
        matched = 0
        mismatched = 0
        violated = 0
        for r in range(self.row_count):
            for c in range(self.col_count):
                if self.board[r][c] == self.board_answer[r][c]:
                    matched += 1
                else:
                    mismatched += 1

        for r in range(self.row_count):
            row = self.__get_row(self.board, r)
            row_status = self.__encode_list(row)
            if row_status != self.row_clues[r]:
                violated += 1
        
        for c in range(self.col_count):
            col = self.__get_col(self.board, c)
            col_status = self.__encode_list(col)
            if col_status != self.col_clues[c]:
                violated += 1
        
        return CheckInfo(matched, mismatched, violated)
    
    def set_cell(self, row_num, col_num, value) -> None:
        self.board[row_num][col_num] = value

    def __get_row(self, board, row_num) -> list:
        return [board[row_num][c] for c in range(self.col_count)]
    
    def __get_col(self, board, col_num) -> list:
        return [board[r][col_num] for r in range(self.row_count)]

    def __encode_list(self, target_list) -> list:
        clue = []
        cnt = 0
        for item in target_list:
            if item == 1:
                cnt += 1
            elif cnt:
                clue.append(cnt)
                cnt = 0
        if cnt:
            clue.append(cnt)
        return tuple(clue)
    
    def encode_board(self, board):
        encode = []
        for j in range(self.col_count):
            count = 0
            previous = 0
            for i in range(self.row_count):
                if board[i][j] == 1 and previous == 0:
                    encode.append(count)
                    count = 0
                    previous = 1
                elif board[i][j] == 0 and previous == 1:
                    previous = 0
                    count = 0
                    i += 1
                else:
                    count += 1
                    previous = board[i][j]
                
        return encode

    def decode_board(self, board):
        decode = [[0 for _ in range(self.col_count)] for _ in range(self.row_count)]
        cur_in = 0
        for j in range(self.col_count):
            '''
            start = first allowed starting position
            allowed = number of allowed position
            placed = start + encoded_distance % allowed
            '''

            start = 0
            not_allowed = sum(self.col_clues[j]) + len(self.col_clues[j]) # Number of blocks that can't be placed counted from the bottom
            for clue in self.col_clues[j]:
                not_allowed -= clue + 1
                allowed = self.row_count - not_allowed - start - (clue - 1)
                placed = start + board[cur_in] % allowed
                for i in range(clue):
                    decode[placed + i][j] = 1
                
                cur_in += 1
                start = placed + clue + 1

        return decode

    def solve(self):
        population_size = 10

        population = [self.generate_random_solution() for _ in range(population_size)]
        
        for item in population:
            self.board = self.decode_board(item)
            print("encoding: ")
            print(np.array(item))
            self.show_board()


    def generate_random_solution(self):
        solution = [random.randint(0,self.row_count-1) for _ in range(self.col_clues_count)]
        # Ensure that the generated solution satisfies the constraints if needed
        return solution




    

if __name__ == "__main__":
    game = Nonogram(10, 10, 0.1)
    game.show_board()
    game.show_answer()
    game.solve()
    result = game.check()
    print(result)