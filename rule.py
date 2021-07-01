empty = 0
class Rule(object):
    def __init__(self, board, board_size):
        self.board = board
        self.board_size = board_size

    #가능한 수 찾기
    def legal_positions(self, stone, board):
        self.board = board
        empty_points = []
        
        #비어있는 점 리스트 만들기
        for i in range(self.board_size):
            for j in range(self.board_size):
                if self.board[i][j] == 0:
                    empty_points.append((j, i))####

        #비어있는 점에서 불가능한 점 제외시키기
        forbidden_points = self.get_forbidden_points(stone)
        valid_points = [i for i in empty_points if i not in forbidden_points]

        return valid_points
    
    #보드 밖을 넘어가지 않는지 확인
    def is_invalid(self, x, y):
        return (x < 0 or x >= self.board_size or y < 0 or y >= self.board_size)

    #보드에 돌 표시
    def set_stone(self, x, y, stone):
        self.board[y][x] = stone

    #8방향 탐색용
    def get_xy(self, direction):
        list_dx = [-1, 1, -1, 1, 0, 0, 1, -1]
        list_dy = [0, 0, -1, 1, -1, 1, -1, 1]
        return list_dx[direction], list_dy[direction]

    #돌 개수 세기
    def get_stone_count(self, x, y, stone, direction):
        x1, y1 = x, y
        cnt = 1
        for i in range(2):
            dx, dy = self.get_xy(direction * 2 + i)
            x, y = x1, y1
            while True:
                x, y = x + dx, y + dy
                if self.is_invalid(x, y) or self.board[y][x] != stone:###
                    break
                else:
                    cnt += 1
        return cnt

    #게임 끝났는지 확인
    def is_gameover(self, x, y, stone, board):
        self.board = board
        for i in range(4):
            cnt = self.get_stone_count(x, y, stone, i)
            print(cnt)
            if cnt >= 5:
                return True
        return False

    #육목
    def is_six(self, x, y, stone):
        for i in range(4):
            cnt = self.get_stone_count(x, y, stone, i)
            if cnt > 5:
                return True
        return False

    #오목
    def is_five(self, x, y, stone):
        for i in range(4):
            cnt = self.get_stone_count(x, y, stone, i)
            if cnt == 5:
                return True
        return False

    #빈 공간 찾기
    def find_empty_point(self, x, y, stone, direction):
        dx, dy = self.get_xy(direction)
        while True:
            x, y = x + dx, y + dy
            if self.is_invalid(x, y) or self.board[y][x] != stone:###
                break
        if not self.is_invalid(x, y) and self.board[y][x] == empty:###
            return x, y
        else:
            return None

    #금수1
    def open_three(self, x, y, stone, direction):
        for i in range(2):
            coord = self.find_empty_point(x, y, stone, direction * 2 + i)
            if coord:
                dx, dy = coord
                self.set_stone(dx, dy, stone)
                if 1 == self.open_four(dx, dy, stone, direction):
                    if not self.forbidden_point(dx, dy, stone):
                        self.set_stone(dx, dy, empty)
                        return True
                self.set_stone(dx, dy, empty)
        return False

    #금수2
    def open_four(self, x, y, stone, direction):
        if self.is_five(x, y, stone):
            return False
        cnt = 0
        for i in range(2):
            coord = self.find_empty_point(x, y, stone, direction * 2 + i)
            if coord:
                if self.five(coord[0], coord[1], stone, direction):
                    cnt += 1
        if cnt == 2:
            if 4 == self.get_stone_count(x, y, stone, direction):
                cnt = 1
        else:
            cnt = 0
        return cnt

    def four(self, x, y, stone, direction):
        for i in range(2):
            coord = self.find_empty_point(x, y, stone, direction * 2 + i)
            if coord:
                if self.five(coord[0], coord[1], stone, direction):
                    return True
        return False

    def five(self, x, y, stone, direction):
        if 5 == self.get_stone_count(x, y, stone, direction):
            return True
        return False

    #금수3
    def double_three(self, x, y, stone):
        cnt = 0
        self.set_stone(x, y, stone)
        for i in range(4):
            if self.open_three(x, y, stone, i):
                cnt += 1
        self.set_stone(x, y, empty)
        if cnt >= 2:
            return True
        return False

    #금수4
    def double_four(self, x, y, stone):
        cnt = 0
        self.set_stone(x, y, stone)
        for i in range(4):
            if self.open_four(x, y, stone, i) == 2:
                cnt += 2
            elif self.four(x, y, stone, i):
                cnt += 1
        self.set_stone(x, y, empty)
        if cnt >= 2:
            return True
        return False

    #금수판별
    def forbidden_point(self, x, y, stone):
        if self.is_five(x, y, stone):
            return False
        elif self.is_six(x, y, stone):
            return True
        elif self.double_three(x, y, stone) or self.double_four(x, y, stone):
            return True
        return False

    #금수 리스트 반환
    def get_forbidden_points(self, stone):
        coords = []
        for y in range(len(self.board)):
            for x in range(len(self.board[0])):
                if self.board[y][x]:####
                    continue
                if self.forbidden_point(x, y, stone):
                    coords.append((x, y))
        return coords
