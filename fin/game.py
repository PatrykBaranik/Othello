import numpy as np

def move(x, y, b, s):
    any = 0
    if b[y, x] != 0:
        return np.array(-100)

    if y > 1:
        if b[y - 1, x] == -1:
            for i in reversed(range(0, y)):
                if b[i, x] == -1:
                    continue
                if b[i, x] == 0:
                    break
                if b[i, x] == 1:
                    for j in range(i, y+1):
                        b[j, x] = 1
                    any = 1
                    break
    ####
    if y < s - 2:
        if b[y + 1, x] == -1:
            for i in range(y + 2, s):
                if b[i, x] == -1:
                    continue
                if b[i, x] == 0:
                    break
                if b[i, x] == 1:
                    for j in range(y, i+1):
                        b[j, x] = 1
                    any = 1
                    break
    ####
    if x > 1:
        if b[y, x - 1] == -1:
            for i in reversed(range(0, x)):
                if b[y, i] == -1:
                    continue
                if b[y, i] == 0:
                    break
                if b[y, i] == 1:
                    for j in range(i, x+1):
                        b[y, j] = 1
                    any = 1
                    break
    ####
    if x < s - 2:
        if b[y, x + 1] == -1:
            for i in range(x + 2, s):
                if b[y, i] == -1:
                    continue
                if b[y, i] == 0:
                    break
                if b[y, i] == 1:
                    for j in range(x, i+1):
                        b[y, j] = 1
                    any = 1
                    break
    ####
    if y > 1 and x > 1:
        if b[y - 1, x - 1] == -1:
            for i in (range(1, min(x, y) + 1)):
                if b[y - i, x - i] == -1:
                    continue
                if b[y - i, x - i] == 0:
                    break
                if b[y - i, x - i] == 1:
                    for j in range(0, i + 1):
                        b[y - j, x - j] = 1
                    any = 1
                    break

    ####
    if y > 1 and x < s - 2:
        if b[y - 1, x + 1] == -1:
            for i in (range(1, min(s - x - 1, y) + 1)):
                if b[y - i, x + i] == -1:
                    continue
                if b[y - i, x + i] == 0:
                    break
                if b[y - i, x + i] == 1:
                    for j in range(0, i + 1):
                        b[y - j, x + j] = 1
                    any = 1
                    break

    ####
    if y < s - 2 and x > 1:
        if b[y + 1, x - 1] == -1:
            for i in (range(1, min(x, s - y - 1) + 1)):
                if b[y + i, x - i] == -1:
                    continue
                if b[y + i, x - i] == 0:
                    break
                if b[y + i, x - i] == 1:
                    for j in range(0, i + 1):
                        b[y + j, x - j] = 1
                    any = 1
                    break

    ####
    if y < s - 2 and x < s - 2:
        if b[y + 1, x + 1] == -1:
            for i in (range(1, min(s - x - 1, s - y - 1) + 1)):
                if b[y + i, x + i] == -1:
                    continue
                if b[y + i, x + i] == 0:
                    break
                if b[y + i, x + i] == 1:
                    for j in range(0, i + 1):
                        b[y + j, x + j] = 1
                    any = 1
                    break

    if any == 0:
        return np.array(-100)
    else:
        return b


class game(object):
    def __init__(self,d):
        board = np.zeros((d,d))
        board[int(d/2),int(d/2)] = -1
        board[int(d / 2-1), int(d / 2-1)] = -1
        board[int(d / 2-1), int(d / 2)] = -1
        board[int(d / 2-2), int(d / 2)] = -1
        board[int(d / 2), int(d / 2-1)] = 1
        self.startBoard = board
        self.board = board
        self.size = d
        self.player = 1
        self.end = 0

    def show(self):
        return(self.board)

    def moved(self,x,y):
        b = move(x,y,self.board*self.player,self.size)
        if b.sum() > -100:
            self.board = b*self.player
            self.player *= (-1)
            self.end = 0
            return np.array(0)
        else:
            return np.array(-100)


    def skip(self):
        self.player *= (-1)
        if self.end == 1:
            self.end = 2
            res = self.board.sum() * self.player
            return res
        self.end = 1

        return 0
    def ai(self,v):

        if v<self.size*self.size:
            return int(self.moved(int(v%self.size), int(v/self.size)))
        else:
            return int(self.skip())

    def posMoves(self):
        pos = []
        for i in range(self.size**2):
            pos.append(max(move(int(i%self.size), int(i/self.size), self.board.copy()*self.player, self.size).flatten())-1)
        if (max(pos)==-101):
            pos.append(0)
        else:
            pos.append(-100)

        return pos


    def isend(self):
        if self.end==2:
            return 1
        else:
            return 0

    def reset(self):
        self.end = 0
        self.board = self.startBoard
        self.player = 1

    def playerNum(self):
        return self.player