### Black pieces are -1 and White are +1. Empty spot is 0


class Board():
    directions = [(1,1),(1,0),(1,-1),(0,-1),(-1,-1),(-1,0),(-1,1),(0,1)]

    def __init__(self, n):
        self.n = n
        self.pieces = [None]*self.n

        for i in range(self.n):
            self.pieces[i] = [0]*self.n

        # Set up the initial 4 pieces.
        self.pieces[int(self.n/2)-1][int(self.n/2)] = 1
        self.pieces[int(self.n/2)][int(self.n/2)-1] = 1
        self.pieces[int(self.n/2)-1][int(self.n/2)-1] = -1
        self.pieces[int(self.n/2)][int(self.n/2)] = -1

     # add [][] indexer syntax to the Board
    def __getitem__(self, i): 
        return self.pieces[i]
    
    def count(self, colour):
        c=0
        for i in range(self.n):
            for j in range(self.n):
                if(self[i][j]==colour):
                    c+=1
                elif(self[i][j]==-colour):
                    c-=1
        return(c)
    
    def legal_moves(self, colour):
        moves = set()
        for i in range(self.n):
            for j in range(self.n):
                if(self[i][j]==colour):
                    m = self.generate_moves((i, j))
                    moves.update(m)

        return(list(moves))

    def check_legal_move(self, colour):
        for i in range(self.n):
            for j in range(self.n):
                if(self[i][j]==colour):
                    m = self.generate_moves((i, j))
                    if(len(m)>0):
                        return True
                    else:
                        return False
                    
    def generate_moves(self, coords):
        (i, j) = coords
        colour = self[i][j]

        if(colour==0):  #empty
            return None
        
        else:
            moves=[]
            for k in self.directions:
                m = self._discover_move(coords, k)
                if(m):
                    moves.append(m)
            return(moves)
        
    def execute(self, move, colour):
        flips = [flip for direction in self.directions
             for flip in self._get_flips(move, direction, colour)]
        if(len(list(flips))>0):
            for i, j in flips:
                self[i][j]=colour

    
    def _discover_move(self, coords, direction):
        x,y = coords
        colour = self[x][y]
        flips = []
        for i, j in self._increment_move(coords, direction, self.n):
            if(self[i][j]==0):
                if(flips):
                    return (i, j)
                else:
                    return
            elif self[i][j]==colour:
                return
            elif self[i][j]==-colour:
                flips.append((i, j))
    
    def _get_flips(self, coords, direction, colour):
        flips = [coords]

        for i, j in self._increment_move(coords, direction, self.n):
            if(self[i][j]==0):
                return []
            if(self[i][j]==-colour):
                flips.append((i, j))

            elif(self[i][j]==colour and len(flips)>0):
                return(flips)
            
        return []
    
    @staticmethod
    def _increment_move(move, direction, n):
        move = list(map(sum, zip(move, direction)))
        while all(map(lambda x: 0 <= x < n, move)):
            yield move
            move = list(map(sum, zip(move, direction)))



        


