import rule
import random
import math
import numpy as np
from copy import deepcopy


def softmax(x):
    probs = np.exp(x - np.max(x))
    probs /= np.sum(probs)
    return probs


class Node:
    """노드 정의
    """

    # board_size 는 new_train에서 받아옴.
    def __init__(self, proir_p, state, board_size, parent_node=None):
        """init

        Args:
            n_parent (int): 부모노드의 방문횟수(= 형제노드들의 방문횟수 총합)
            state (list): 오목판 현재 상태
            parent_node (node, optional): 부모노드. 루트노드일 경우 None.
        """
        self.parent = parent_node  # 부모노드, 이전 상태
        self.children = {}  # 자식노드, 다음 상태

        self.board_size = board_size
        self.clear_board = [[0 for i in range(self.board_size)] for j in range(
            self.board_size)]  # new_train과 맞지 않을 수 있는 변수 제거

        #가중치 계산에 필요한 값
        self.n = 0  # 방문횟수
        self.Q = 0  # win/lose/tie 노드 평가
        self.u = 0  # 방문횟수에 따라 조정된 값
        self.p = proir_p

        self.state = state  # 현재 보드 상태
        self.x = 0  # x좌표
        self.y = 0  # y좌표
        self.stone = 0  # 흑/백

        self.get_xystone()  # state에서 x, y, stone 값 찾기

    def get_value(self, c):
        """가중치 계산

        Args:
            c (float): exploration / exploitation

        Returns:
            float: 계산된 가중치
        """
        self.u = (c * self.p * np.sqrt(self.parent.n) / (1 + self.n))
        return self.Q + self.u

    def get_xystone(self):
        """x, y, stone 값 찾기
        """
        if self.parent == None:
            self.x = 7
            self.y = 7
            self.stone = 1
            return 

        for i in range(self.board_size):
            for j in range(self.board_size):
                #부모노드에서의 보드 상태와 자식노드에서의 보드 상태를 비교해서 다른 곳이 이번에 둔 곳
                if self.parent.state[i][j] != self.state[i][j]:
                    self.x = j
                    self.y = i
                    self.stone = self.state[i][j]

    def isleaf(self):
        """리프노드 판별

        Returns:
            bool: 리프노드 판별
        """
        return self.children == {}

    def isroot(self):
        """루트노드 판별

        Returns:
            bool: 루트노드 판별
        """

        return self.parent == None


class MCTS:
    """MCTS 정의
    """

    # board_size 는 new_train에서 받아옴.
    def __init__(self, board, board_size, policy_value, c=1.0, temp=1e-3):
        """init

        Args:
            board (list): 현재 보드 상태
            policy_value (func): 신경망에서 (state, policy) 받아옴.
            c (float, optional): exploration / exploitation Defaults to 1.
        """

        self.state = board
        self.board_size = board_size
        self.policy_value = policy_value
        self.c = c
        self.temp = temp

        self.clear_board = [[0 for i in range(self.board_size)] for j in range(self.board_size)]  # new_train과 맞지 않을 수 있는 변수 제거

        #Q값 계산하기 위한 값 (승패에 따라 달라짐)
        self.winpoint = 1.0
        self.losepoint = -1.0
        self.tiepoint = 0

        self.rule = rule.Rule(board=self.state, board_size=self.board_size)

        self.black = 1
        self.white = 2
    
    def record_board(self):
        """오목 끝난 후 보드 기록
        """
        try :
            f = open('boardRecords.txt', 'r+')
        
        except FileNotFoundError :
            f = open('boardRecords.txt', 'w+')
        
        f.readlines()

        for i in self.state[:]:
            for j in i[:]:
                f.write('%d' % j)
                f.write(' ')

            f.write('\n')

        f.write('-----------------------------\n')
        f.close()

    def expand(self, current_node, policy):
        """확장

        Args:
            current_node (Node): 현재 노드
            policy : 신경망에서 받아온 값
        """
        for action, prob in policy: #action = 좌표, prob = p
            board = deepcopy(self.state) #다음 상태 기록용 보드
            if action not in current_node.children:
                board[action[1]][action[0]] = current_node.stone + 1 if current_node.stone == 1 else current_node.stone - 1 #보드에 좌표 찍기 (흑 = 1, 백 = 2)
                current_node.children[action] = Node(prob, board, self.board_size, current_node) #자식노드에 추가

    def select(self, current_node):
        """선택

        Args:
            current_node (Node): 현재 노드

        Returns:
            [tuple]: (action,node) 
        """
        #가중치 리스트 만들기
        weights = []
        for i in current_node.children.values():
            weights.append(i.get_value(self.c))

        try:
            return random.choices(list(current_node.children.items()), weights=weights)
        except ValueError: #가중치합이 0 이하일 경우 에러 -> 가중치 없이 그냥 랜덤
            return random.choice(list(current_node.children.items()))
        # return max(current_node.children.items(), key=lambda node: node[1].get_value(self.c).any())

    def form_state(self, playlen, last_node):
        """신경망 학습용 보드로 변환

        Args:
            playlen (int): 플레이 길이
            last_node (Node): 마지막 노드(리프 노드)

        Returns:
            [array]: [3차원 보드]
        """
        #비어있는 3차원 array 만든 후 해당 값만 채우기
        #0층 : 흑 (1, 0)
        #1층 : 백 (1, 0)
        #2층 : 마지막으로 둔 곳 (1, 0)
        #3층 : 마지막으로 둔 돌 (흑 1, 백 0)
        square_state = np.zeros((4, self.board_size, self.board_size))
        for i in range(self.board_size):
            for j in range(self.board_size):
                if self.state[i][j] != 0:
                    if self.state[i][j] == (playlen % 2) + 1: 
                        square_state[0][i][j] = 1.0
                    else:
                        square_state[1][i][j] = 1.0
        square_state[2][last_node.x][last_node.y] = 1.0
        if playlen % 2 == 0:
            square_state[3][:][:] = 1.0
        return square_state

    def form_probs(self, current_node):
        """신경망 학습용 probs map

        Args:
            current_node (Node): 현재 노드

        Returns:
            [array]: [2차원 리스트]
        """
        #비어있는 2차원 array 만든 후 해당 값만 채우기
        #자식노드에 있는 것만 값 할당( 가능한 수가 아니라면 확률은 0이므로 )
        move_probs = np.zeros((self.board_size, self.board_size))
        for i in current_node.children.values():
            move_probs[i.x][i.y] = i.n
        return softmax(1.0/self.temp * np.log(move_probs + 1e-10))

    def form_winner(self, winner, playlen, playlog_node):
        """신경망 학습용 winner list

        Args:
            winner (int): 이긴 돌의 색, 비겼을 경우 -1
            playlen (int): 플레이 길이
            playlog_node (list): 플레이 동안 선택한 노드의 리스트

        Returns:
            [array]: [1차원 리스트]
        """
        #비어있는 1차원 array 만든 후 채우기
        #이기면 1, 지면 -1, 비기면 0
        winner_z = np.zeros(playlen)

        if winner == -1:
             return winner_z

        for i in range(playlen):
            if playlog_node[i].stone == winner: #winner랑 돌 같으면 win이니까 1
                winner_z[i] == 1.0
            else: #아니면 lose라서 -1
                winner_z[i] == -1.0

        return winner_z

    def simulation(self, root_node):
        """시뮬레이션 : 실제 플레이

        Args:
            root_node (Node): 루트 노드 (7,7) stone = 흑

        Returns:
            [tuple]: [playdata = board + probs + winner]
        """
        current_node = root_node
        playlen = 0 #플레이 길이
        playlog_node = [] #플레이 동안 선택한 노드의 리스트
        playlog_board = [] #플레이 동안의 모든 오목판 상태
        state_data = [] #신경망 학습용 보드 리스트
        probs_data = [] #신경망 학습용 probs map 리스트

        while True:
            self.state = current_node.state #이전 보드에서 현재 보드로 바꾸기
            playlog_node.append(current_node) #노드 기록
            playlog_board.append(self.state) #보드 기록
            playlen += 1
            print(self.state) # debugging
            state_data.append(self.form_state(playlen, current_node)) #신경망 학습용으로 변환하여 데이터 추가

            if current_node.children == {}: #자식노드 없으면 확장
                legal_positions = self.rule.legal_positions(current_node.stone, self.state) #rule에서 가능한 수 받아오기
                policy, value = self.policy_value(self.form_state(playlen, current_node), current_node.stone, legal_positions) #신경망에서 prob 값 받아오기
                self.expand(current_node, policy) #확장

            probs_data.append(self.form_probs(current_node)) #신경망 학습용 probs map 데이터 추가

            #게임 끝났는지 확인
            if self.rule.is_gameover(current_node.x, current_node.y, current_node.stone, self.state):
                self.record_board() #텍스트 파일에 결과 기록
                winners_data = self.form_winner(
                    current_node.stone, playlen, playlog_node) #신경망 학습용 winner 데이터 추가
                self.update(playlog_node, winplayer=current_node.stone) #가중치 갱신

                print("Game over. Winner is {}".format(
                    current_node.stone))  # 결과 출력

                self.state = deepcopy(self.clear_board) #보드 초기화
                self.rule = rule.Rule(self.state, self.board_size) #rule 초기화
                # play_data 자료형 list -> tuple(zip)
                play_data = zip(state_data, probs_data, winners_data)
                return play_data

            #확장 후 children이 비어있다 -> 가능한 수가 없다 -> 비겼다
            if current_node.children == {}:
                self.record_board()  # 텍스트 파일에 결과 기록
                winners_data = self.form_winner(-1, playlen, playlog_node) #신경망 학습용 winner 데이터 추가
                self.update(playlog_node, tie=1)  # 가중치 갱신

                print("Game over. Winner is nobody.")  # 결과 출력

                self.board = deepcopy(self.clear_board)  # 보드 초기화
                self.rule = rule.Rule(self.state, self.board_size)  # rule 초기화
                # play_data 자료형 list -> tuple(zip)
                play_data = zip(state_data, probs_data, winners_data)
                return play_data

            next_move = self.select(current_node) #다음 수 선택
            if type(next_move) == tuple:
                next_move = [next_move] #튜플로 들어오는 값 리스트로 감싸주기
            current_node = next_move[0][1]


    def update(self, playlog, winplayer=1, tie=0):
        """가중치 갱신

        Args:
            playlog (list): 플레이 동안에 선택한 노드들의 리스트
            winplayer (int, optional): [흑 / 백, 비겼을 때를 대비해 디폴트값 설정]. Defaults to 1.
            tie (int, optional): [비겼을 때 1, 아니면 0]. Defaults to 0.
        """
        if not tie: #비기지 않았을 경우
            for i in playlog: #플레이 로그에 기록된 모든 노드에 가중치 갱신
                if i.stone == winplayer:
                    i.n += 1
                    i.Q += (self.winpoint - i.Q) / i.n
                else:
                    i.n += 1
                    i.Q += (self.losepoint - i.Q) / i.n
        else: #비긴 경우
            for i in playlog: #플레이 로그에 기록된 모든 노드에 가중치 갱신
                i.n += 1
                i.Q += (self.tiepoint - i.Q) / i.n
