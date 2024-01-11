import pygame
import sys

from const import *
from game import Game
from square import Square
from move import Move

class Main:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH,HEIGHT))
        pygame.display.set_caption('Chess')
        self.game = Game()

    def mainloop(self):
        
        game = self.game
        screen = self.screen
        board = self.game.board
        ai = self.game.ai
        dragger = self.game.dragger

        while True:
            if not game.selected_piece:
                game.show_bg(screen)
                game.show_pieces(screen)

            game.show_hover(screen)

            if dragger.dragging:
                dragger.update_blit(screen)

            for event in pygame.event.get():
                
                # click
                if event.type == pygame.MOUSEBUTTONDOWN:
                    # for row in range(ROWS):
                    #     for col in range(COLS):
                    #         print(game.board.squares[row][col].piece,end='')
                    #     print()
                    dragger.update_mouse(event.pos)

                    pos = event.pos
                    clicked_row = dragger.mouseY // SQSIZE
                    clicked_col = dragger.mouseX // SQSIZE

                    # if clicked square has a piece ?
                    if Square.in_range(clicked_col,clicked_row):
                        if board.squares[clicked_row][clicked_col].has_piece():
                            piece = board.squares[clicked_row][clicked_col].piece
                            #valid piece (color) ?
                            if piece.color == game.next_player:
                                game.select_piece(piece)
                                board.calc_moves(piece, clicked_row, clicked_col, bool=True)
                                dragger.save_initial(event.pos)
                                dragger.drag_piece(piece)
                                #show methods
                                game.show_bg(screen)
                                game.show_pieces(screen)

                # mouse motion
                elif event.type == pygame.MOUSEMOTION:
                    

                    motion_row = event.pos[1] // SQSIZE
                    motion_col = event.pos[0] // SQSIZE

                    if Square.in_range(motion_row,motion_col):
                        game.set_hover(motion_row,motion_col)

                    if dragger.dragging:
                        dragger.update_mouse(event.pos)
                        #show methods
                        game.show_bg(screen)
                        game.show_pieces(screen)
                        game.show_hover(screen)
                        dragger.update_blit(screen)

                # click release
                elif event.type == pygame.MOUSEBUTTONUP:

                    if dragger.dragging:
                        dragger.update_mouse(event.pos)

                        released_row = dragger.mouseY // SQSIZE
                        released_col = dragger.mouseX // SQSIZE

                        #create possible move
                        initial = Square(dragger.initial_row,dragger.initial_col)
                        final = Square(released_row,released_col)
                        move = Move(initial,final)

                        # valid move ?
                        if board.valid_move(dragger.piece,move):
                            # normal capture
                            captured = board.squares[released_row][released_col].has_piece()
                            board.move(dragger.piece, move)
                            # sounds
                            game.play_sound(captured)

                            board.set_true_en_passant(dragger.piece)

                            # show methods
                            game.show_bg(screen)
                            # game.show_last_move(screen)
                            game.show_pieces(screen)
                            # next -> AI
                            game.next_turn()

                            # --------------
                            # >>>>> AI >>>>>
                            # --------------

                            if game.gamemode == 'ai':
                                # update
                                game.unselect_piece()
                                game.show_pieces(screen)
                                pygame.display.flip()
                                #optimal move
                                move = ai.eval(board)
                                initial = move.initial
                                final = move.final
                                # piece
                                piece = board.squares[initial.row][initial.col].piece
                                # capture
                                captured = board.squares[final.row][final.col].has_piece()
                                # move
                                board.move(piece, move)
                                game.play_sound(captured)
                                # draw
                                game.show_bg(screen)
                                game.show_pieces(screen)
                                # next -> AI
                                game.next_turn()
                            

                    game.unselect_piece()
                    dragger.undrag_piece()

                # key press
                elif  event.type == pygame.KEYDOWN:

                    # changing themes
                    if event.key == pygame.K_t:
                        game.change_theme()
                    
                    # gamemode
                    if event.key == pygame.K_a:
                        game.change_gamemode()

                    # restarting the game
                    if event.key == pygame.K_r:
                        game.reset()

                        game = self.game
                        board = self.game.board
                        dragger = self.game.dragger
                        ai = self.game.ai

                    # depth
                    if event.key == pygame.K_3:
                        ai.depth = 2

                    if event.key == pygame.K_4:
                        ai.depth = 3
                    
                # quit the application
                elif event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
            

            pygame.display.flip()


main = Main()
main.mainloop()