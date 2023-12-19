import random
import copy
from collections import namedtuple

import numpy as np
import random
import matplotlib.pyplot as plt

CheckInfo = namedtuple('CheckInfo', ['matched', 'mismatched', 'violated'])

class Nonogram(object):
    def __init__(self, row_count=10, col_count=10, fill_rate=0.1,
                 assigned_answer=None,
                 population_size=10, max_generation=5, crossover_prob=0.9, mutation_prob=0.1):
        """
        Args:
            row: Number of rows for the board.
            col: Number of columns for the board.
            fill_rate: Rate of filled cells.
            assigned_answer: list of numbers (in [0, row_count * col_count]). 
                If assigned, the board answer will become assigned answer.
                else, random generate by sample(total, total*fill_rate)

            ----
            population size: size of populations
            max generations: number of generations
            crossover prob: if a random number smaller than this threshold, perform self.crossover
            mutation prob: if a random number smaller than this threshold, perform self.mutation
        """
        
        self.row_count = row_count
        self.col_count = col_count
        self.total_count = self.row_count * self.col_count
        self.fill_rate = fill_rate
        
        self.board = [ [0 for _ in range(col_count)] for _ in range(row_count)]
        self.board_answer = copy.deepcopy(self.board)

        self.row_clues = []
        self.col_clues = []
        self.col_clues_count = 0

        # for genetic algorithms
        self.populations = None
        self.population_size = population_size
        self.max_generation = max_generation
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.max_fitness = np.zeros(self.max_generation + 1)

        self.recorded_max_fitness = []

        # assign answer if assigned in parameters
        # if not, random generated
        # Initialize board layout

        if(assigned_answer is None):
            samples = random.sample(range(self.total_count), int(self.total_count * self.fill_rate))
        else:
            samples = assigned_answer
        
        for sample in samples:
            self.board_answer[sample // self.col_count][sample % self.col_count] = 1

        self.row_clues = [] * self.row_count
        self.col_clues = [] * self.col_count
        
        for r in range(self.row_count):
            row = self.__get_row(self.board_answer, r)
            self.row_clues.append(self.__encode_list(row))
        
        for c in range(self.col_count):
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
            if len(self.row_clues[i]) == 0: output += "0"
            output = output + "\n"
        for i in range(self.col_count): 
            output += "- "
        output += "\n"
        for j in range(max([len(item) for item in self.col_clues])):
            for i in range(self.col_count):
                if len(self.col_clues[i]) > j: 
                    output += str(self.col_clues[i][j])
                else:
                    if j == 0:
                        output += "0"
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
            if len(self.row_clues[i]) == 0: output += "0"
            output = output + "\n"
        for i in range(self.col_count): 
            output += "- "
        output += "\n"
        for j in range(max([len(item) for item in self.col_clues])):
            for i in range(self.col_count):
                if len(self.col_clues[i]) > j: 
                    output += str(self.col_clues[i][j])
                else:
                    if j == 0:
                        output += "0"
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
    
    def calculate_fitness(self, population) -> float:
        """helper function for fitness"""
        self.board = self.decode_board(population)
        return float(self.fitness())

    def fitness(self) -> int:
        """
        fitness = -1 * sum( delta(genetic row clue, answer row clue))
        the more candidate matches the target row clue, the less fitness value get

        for instacne:
        row   correct wrong     delta
        0     [3]     [3,1]     |3-3| + |0-1| = 1
        1     [2,1]   [2,1,1]   |2-2| + |1-1| + |0-1| = 1

        fitness = -sum(delta_i) = -(1+1) = -2
        """
        # get decode rules      
        decoded_clue = [[] for _ in range(self.row_count)]
        for row in range(self.row_count):
            current_count = 0
            is_counting = False

            for col in range(self.col_count):
                if(self.board[row][col] == 1):
                    is_counting = True
                    current_count += 1
                    # go to next row, push remaining count to list
                    if(col == self.col_count - 1):
                        decoded_clue[row].append(current_count)

                elif(is_counting):
                        decoded_clue[row].append(current_count)
                        current_count = 0
                        is_counting = False

        # calculate delta(row clue, decode clue)
        delta_sum = 0
        for r in range(self.row_count):
            decoded_c = decoded_clue[r]
            target_c = self.row_clues[r]

            # decide smaller length
            shorter, longer = [], []
            if(len(decoded_c) < len(target_c)):
                shorter = decoded_c
                longer = target_c
            else:
                shorter = target_c
                longer = decoded_c

            i = 0
            for _ in range(len(shorter)):
                delta_sum += abs(shorter[i] - longer[i])
                i += 1
            for _ in range(len(longer) - len(shorter)):
                delta_sum += abs(longer[i])
                i += 1

        return self.row_count * self.col_count -1 * delta_sum

    def record_fitness(self):
        max_fit = max([self.calculate_fitness(p) for p in self.populations])
        self.recorded_max_fitness.append(max_fit)
        return max_fit

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

    def roulette_selection(self):
        # selection: roulette wheel selection with replacement
        total_fits = sum([self.calculate_fitness(p) for p in self.populations])
        selection_prob = [self.calculate_fitness(p) / float(total_fits) for p in self.populations]
        selected_index = np.random.choice(len(self.populations), len(self.populations), 
                                              p=selection_prob)

        # replacement: generational model (no elitsim)
        return self.populations[ selected_index ]
    
        # only implement size=2...
    
    def tournament_selection(self):
        selected_index = []
        for _ in range(self.population_size):
            parents_idx = np.random.choice(len(self.populations), size=2)
            larger = parents_idx[0]
            if( self.calculate_fitness(self.populations[parents_idx[0]]) < 
                self.calculate_fitness(self.populations[parents_idx[1]]) ):
                larger = parents_idx[1]
            selected_index.append(larger)

        selected_index = np.array(selected_index)
        self.populations = self.populations[ selected_index ]

    def crossover(self):
        # single-point crossover, with prob to determine whether to happen
        happen = random.randint(0, 1000) / 1000.0
        if(happen > self.crossover_prob):
            return
        
        # for each consecutive pair..
        for i in range(0, self.population_size, 2):
            # swap based on column clue
            crossover_col = random.randint(1, len(self.col_clues)-2)
            crossover_point = 0
            for j in range(crossover_col):
                crossover_point += len(self.col_clues[j])
            
            # swap segment..
            tmp = self.populations[i][crossover_point:].copy()
            self.populations[i][crossover_point:] = self.populations[i+1][crossover_point:]
            self.populations[i+1][crossover_point:] = tmp

    def mutation(self):
        for i in range(self.population_size):
            # determine perform or not
            happen = random.randint(0, 1000) / 1000.0
            if(happen > self.mutation_prob):
                continue

            mutation_point = random.randint(0, self.col_clues_count-1)
            self.populations[i][mutation_point] = random.randint(0,self.row_count-1)           

    def run_generations(self):
        # including gen 0
        self.max_fitness[0] = self.record_fitness()

        for g in range(self.max_generation):
            self.roulette_selection()
            self.crossover()
            self.mutation()

            # new generation
            self.max_fitness[g+1] = self.record_fitness()
            
            print("iter " + str(g+1))
            print(self.max_fitness[g+1])
            # print("encoding:")
            # for item in self.populations:
            #     print(np.array(item))

        return self.max_fitness
    
    def plot_result(self):
        xs = list(range(self.max_generation + 1))
        plt.plot(xs, self.recorded_max_fitness)
        plt.xlabel("time (generation)")
        plt.ylabel("averaged fitness value")
        plt.title("Result graph")
        plt.savefig("nonogram_result.png")
        plt.show()

    def solve(self):
        # initial generation
        self.populations = np.array([self.generate_random_solution() for _ in range(self.population_size)])

        # run genetics
        self.run_generations()

        self.plot_result()

        
    def generate_random_solution(self):
        solution = [random.randint(0,self.row_count-1) for _ in range(self.col_clues_count)]
        # Ensure that the generated solution satisfies the constraints if needed
        return solution  

if __name__ == "__main__":
    # this is the showcase in https://github.com/morinim/vita/wiki/nonogram_tutorial
    game = Nonogram(row_count=9, col_count=8,
        assigned_answer=[
            1,2,3,
            8,9,11,
            17,18,19,22,23,
            26,27,30,31,
            34,35,36,37,38,39,
            40,42,43,44,45,46,
            48,49,50,51,52,53,
            60,
            67,68
    ],
    population_size=100, max_generation=int(1e3),
    mutation_prob=0.05)
    game.show_board()
    game.show_answer()
    
    game.solve()
    # result = game.check()
    # print(result)