## EXTERNAL DEPENDENCIES
import pandas as pd 
from psychopy import visual
import os

## INTERNAL DEPENDENCIES
import fx

## RETRIEVE PARAMETERS FROM .CSV FILE
previous_parameters = pd.read_csv('parameters.csv')
current_parameters, functions = fx.prompt(previous_parameters)

## UPDATE PARAMETER .CSV FOR NEXT RUN

# bring currently selected function/problem to top
functions.remove(current_parameters.data[0])
functions.insert(0, current_parameters.data[0])

problems = ['shj1','shj2','shj3','shj4','shj5','shj6','xor','circle-in-square','iris','discotypes','loops1']
problems.remove(current_parameters.data[1])
problems.insert(0, current_parameters.data[1])

# update/overwrite old .csv
previous_parameters.loc[:len(functions)-1,'fx'] = functions
previous_parameters.loc[:,'problem'] = problems
previous_parameters.loc[:,'lr1'] = float(current_parameters.data[2])
previous_parameters.loc[:,'lr2'] = float(current_parameters.data[3])
previous_parameters.loc[:,'in_wtr'] = float(current_parameters.data[4])
previous_parameters.loc[:,'n_hids'] = int(current_parameters.data[5])
previous_parameters.loc[:,'n_epochs'] = int(current_parameters.data[6])
previous_parameters.loc[:,'n_itr'] = int(current_parameters.data[7])
previous_parameters.loc[:,'max'] = int(current_parameters.data[8])
previous_parameters.loc[:,'c'] = float(current_parameters.data[9])
previous_parameters.loc[:,'map'] = float(current_parameters.data[10])

previous_parameters.to_csv(os.path.join(os.getcwd(), 'parameters.csv'), index=False)

## SPECIFY DATA SET AND MODEL TO RUN IT
problem = current_parameters.data[1]
full_set, n_classes, n_dims = fx.preprocessing(problem)
act_fx, wt_gen, wt_update = fx.run_fx[str(previous_parameters['fx'][0])]

## RUN SPECIFIED MODEL
run_outcome, output_activations, wts = fx.run_model(problem, previous_parameters, full_set, n_classes, n_dims, wt_gen, wt_update, act_fx)

## CREATE PSYCHOPY VARIABLES
win = visual.Window(fullscr=True, units='pix', color='#FFFFFF')
text = visual.TextStim(win, wrapWidth = 1000, color='#000000', font='Rockwell', pos=(0, 0), height=28, text = '')
text_2 = visual.TextStim(win, wrapWidth = 1000, color='#000000', font='Rockwell', pos=(0, -100), height=30, text = '')
img = visual.ImageStim(win, pos=(0,0))

## PRESENT RESULTS
fx.draw_and_wait(win, text, run_outcome)

## COLLECT USER RESPONSE, RUN FIXED MODEL
while True:
	user_response = fx.get_item(win, text, text_2, n_dims, fx.key_names)
	if user_response == 'gif':
		wt_history, full_copy, iteration_performance, new_dims = fx.wt_watcher(previous_parameters, full_set, n_classes, n_dims, wt_gen, wt_update, act_fx)
		output_dir = fx.gif_fx[new_dims](win, full_copy, problem, previous_parameters['n_hids'][0], wt_history, iteration_performance, n_classes)
		fx.animate(output_dir, problem)
		fx.draw_and_wait(win, text, 'gif generated. Press SPACE to continue')
		continue
	fx.run_query(win, user_response, wts, previous_parameters, n_classes, n_dims, act_fx, text, text_2, img)



