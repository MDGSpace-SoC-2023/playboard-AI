
class Square:

    ALPHACOLS = { 0 : 'a', 1 : 'b', 2 : 'c', 3 : 'd', 4 : 'e', 5 : 'f', 6 : 'g', 7 : 'h'}
        
    def __init__(self, row, col, piece = None):
        self.row = row
        self.col = col
        self.piece = piece
        self.alphacol = self.ALPHACOLS[col]
    
    def __str__(self):
        return '(' + str(self.row) + ', ' + str(self.col) + ')'

    def __eq__(self, other):
        return self.row == other.row and self.col == other.col
    
    # -------------
    # OTHER METHODS
    # -------------

    def has_piece(self):
        return self.piece != None

    def isempty(self):
        return not self.has_piece()

    def has_team_piece(self,colour):
        return self.has_piece() and self.piece.color == colour

    def has_enemy_piece(self,colour):
        return self.has_piece() and self.piece.color != colour

    def isempty_or_enemy(self,colour):
        return self.isempty() or self.has_enemy_piece(colour)

    @staticmethod
    def in_range(*args):
        for arg in args:
            if arg < 0 or arg > 7:
                return False
        
        return True
    
    @staticmethod
    def get_alphacol(col):
        ALPHACOLS = { 0 : 'a', 1 : 'b', 2 : 'c', 3 : 'd', 4 : 'e', 5 : 'f', 6 : 'g', 7 : 'h'}
        return ALPHACOLS[col]
        
        