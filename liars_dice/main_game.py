import tkinter as tk
from tkinter import font
from tkinter import simpledialog
import random
import torch
import re
from NNET import *
from game import *

class LiarsDiceGame:
    def __init__(self, master):
        self.master = master
        self.master.title("Liar's Dice Game")
        self.master.geometry("500x500")
        self.master.configure(bg='light green')
        self.master.bind("<Up>", self.main_game)
        self.create_widgets()

    def display_text(self,output_text):
        self.message_box.config(state=tk.NORMAL)  # Set state to normal to allow modifications
        self.message_box.insert(tk.END, output_text+ "\n")  # Insert new text
        self.message_box.config(state=tk.DISABLED)  # Set state back to disabled
        self.message_box.see(tk.END)

    def create_widgets(self):
        # Create a frame to center the widgets
        bold_font = font.Font(family="Helvetica", size=25, weight="bold")
        text_font = font.Font(family="Helvetica", size=12)

        center_frame = tk.Frame(self.master, bg='light green', pady=10)
        center_frame.pack(expand=True)

        self.top_label = tk.Label(center_frame, text="99% of gamblers quit right before winning big", font=text_font,pady=0)
        self.top_label.pack()  # Pack it at the bottom with some padding
        self.top_label.configure(bg='light green')

        self.robot_roll_label = tk.Label(center_frame, text="Robot's Roll: \n ? ? ? ? ?", font=bold_font, pady=20)
        self.robot_roll_label.pack()
        self.robot_roll_label.configure(bg='light green')  

        # Create a Text widget for displaying messages
        self.message_box = tk.Text(center_frame, height=5, width=50, state=tk.DISABLED)
        self.message_box.pack(pady=10)
        self.message_box.config(font=text_font)



        self.roll_label = tk.Label(center_frame, text="Your Roll:\n", font=bold_font, pady=20)
        self.roll_label.pack()
        self.roll_label.configure(bg='light green')  

        self.bottom_label = tk.Label(center_frame, text="Press the UP arrow to start/restart the game", font=text_font,pady=0)
        self.bottom_label.pack()  # Pack it at the bottom with some padding
        self.bottom_label.configure(bg='light green')
    
    def update_display(self, r1, r2):
        your_roll_text = ', '.join(map(str, r2))
        robot_roll_text = ', '.join(map(str, r1))

        self.roll_label.config(text=f"Your Roll: {your_roll_text}")
        self.robot_roll_label.config(text=f"Robot's Roll: {robot_roll_text}")


    def main_game(self,event):
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

        d1, d2 = 2, 2
        sides = 6
        path = "./model/model22.zip"

        checkpoint = torch.load(path, map_location=torch.device(DEVICE))


        D_PUB, D_PRI, *_ = arguments(
            d1, d2, sides
        )
        model = Net(D_PRI, D_PUB)
        model.load_state_dict(checkpoint["model_state_dict"])
        game = Game(model, d1, d2, sides)

        class Human:

            def __init__(self, game_instance):
                self.game_instance = game_instance

            def get_action(self, state):
                last_call = game.last_call_state(state)
                while True:
                    user_input = simpledialog.askstring("Input",'Your call [e.g. 24 for 2 fours, or "lie" to call a bluff]: ')
                    call = user_input
                    if call == "lie":
                        return game.lie
                    elif m := re.match(r"(\d)(\d)", call):
                        n, d = map(int, m.groups())
                        action = (n - 1) * game.sides + (d - 1)
                        if action <= last_call:
                            self.game_instance.display_text(f"Can't make that call after {repr_action(last_call)}")
                            #print(f"Can't make that call after {repr_action(last_call)}")
                        elif action >= game.lie:
                            self.game_instance.display_text(f"The largest call you can make is {repr_action(game.lie-1)}")
                            #print(
                            #    f"The largest call you can make is {repr_action(game.lie-1)}"
                            #)
                        else:
                            return action



        class AI:
            def __init__(self, game_instance, priv):
                self.game_instance = game_instance
                self.priv = priv

            def get_action(self, state):
                last_call = game.last_call_state(state)
                return game.sample(self.priv, state, last_call, exp=0)

            def __repr__(self):
                return "robot"


        def repr_action(action):
            action = int(action)
            if action == -1:
                return "nothing"
            if action == game.lie:
                return "lie"
            n, d = divmod(action, game.sides)
            n, d = n + 1, d + 1
            return f"{n} {d}s"
        
        while d1>0 and d2>0:
            
            path = "./model/model" + str(d1) + str(d2) + ".zip"

            checkpoint = torch.load(path, map_location=torch.device(DEVICE))


            D_PUB, D_PRI, *_ = arguments(
                    d1, d2, sides
                )
            model = Net(D_PRI, D_PUB)
            model.load_state_dict(checkpoint["model_state_dict"])
            game = Game(model, d1, d2, sides)

            r1 = random.choice(list(game.roll(0)))
            r2 = random.choice(list(game.roll(1)))
            privs = [game.private(r1, 0), game.private(r2, 1)]
            state = game.public()
            
            self.display_text(f"> You rolled {r1}!")
            your_roll_text = ', '.join(map(str, r1))
            self.roll_label.config(text=f"Your Roll:\n {your_roll_text}")
            # print(f"> You rolled {r1}!")
            players = [Human(self), AI(self, privs[1])]

            cur = 0
            while True:
                action = players[cur].get_action(state)
                play = "human" if cur==0 else "robot"

                self.display_text(f"> The {play} called {repr_action(action)}!")

                if action == game.lie:
                    last_call = game.last_call_state(state)
                    res = game.evaluate(r1, r2, last_call)
                    self.display_text(f"> The rolls were {r1} and {r2}.")
                    if res:
                        self.display_text(f"> The call {repr_action(last_call)} was good!")
                        self.display_text(f"> The {play} loses!")
                        if(cur==0):
                            d1-=1
                        else:
                            d2-=1
                        
                    
                    else:
                        self.display_text(f"> The call {repr_action(last_call)} was a bluff!")
                        self.display_text(f"> The {play} wins!")
                        if(cur==0):
                            d2-=1
                        else:
                            d1-=1
                        
                        
                    if(d1==0):
                        self.display_text(f"> Robot wins the overall game")
                    elif(d2==0):
                        self.display_text(f"> Human wins the overall game")
                    break

                state = game.apply_action(state, action)
                cur = 1 - cur

root = tk.Tk()
game_app = LiarsDiceGame(root)
root.mainloop()


