# Connect-four
This code implements several agents for the game "Connect four", and is written with PyCharm for Ubuntu system, but it should work for all platforms that suppports python. To run this code, you need to install Python3(copy, time and math should be installed at the same time), numpy, matplotlib, and keras. 

To run the game, you should run main.py. You can choose each of the two players from Monte Carlo agent, CNN agent, random agent (make totally random decision), and human agent (the user watches the board, and type in the column he chooses in every of his turn), and type your choice in 28th and 29th row in the code. The parameters that can vary includes the number of board rows R, the number of board columns C, and the number in a line to win win_num, please type your choice in row 14, 15 and 16 in the code. All player except CNN agent accepts all combination of parameters so that min(R, C)>win_num. I trained CNN with these seven combinations and saved in the project for you to try, and you can also train different models with different parameters (training parameters are automatically set, you only need to decide R, C and win_num). These seven CNN options are:

[R, C, win_num]: [8,10,5], [5,10,4], [6,7,4], [10,5,4], [5,5,4], [6,7,3], [5,5,3]

To train CNN agent with different problem instance size, you should modify R, C, win_num and run these 3 files in order:

main_CNN_training_data.py  CNN.py  main.py

All other parameters are automatically set.

For training CNN purpose, you can change search_step to 1000 in file monte_carlo_tree.py, row 67. For shorter runtime, you can change it to 100.

plot.py is for 667 course project report, you can get all these data from the three files:

main_CNN_training_data.py  CNN.py  main.py
