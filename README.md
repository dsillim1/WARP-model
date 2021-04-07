
# INITIAL SET UP BEFORE RUNNING

1. To install the required packages, make sure you have Python 3 installed. Just use your preferred search engine.

2. Use the following command* from a BASH terminal or Powershell (if using Windows). You may want to set up a virtual environment first (though optional):
pip install -r requirements.txt

   1. **NOTE**: as of writing this document, I have personally had issues downloading the latest psychopy via pip.
If you run into these issues, I'd recommend downloading the package straight from the official website. It works the same.


# INSPECTOR OPERATION      

1. Navigate to the WARP inspector directory with terminal/Powershell. Alternatively, use psychopy.

2. If using the former option, run the script by entering either 'py launcher.py' or 'python3 launcher.py' depending on OS.

3. Follow the prompt instructions to select problem, model, and hyperparameters (HPs).

4. A splash page should show the initial results. From this point forward, the escape key should work at any time to exit the program.

5. If you choose to continue, you will be provided an opportunity to probe the current weight space with an input item that you type out.

   1. The columns labeled 'F#' contains values representing where in psychological space that given node resides on that given feature dimension.
For convenience, we assume that the incoming stimulus and the reference points share the same representational vocabulary and are represented via the same mechanisms.

6. Alternatively, you can type 'gif' to create a visualization of how that model, with those HPs, might handle the given problem.

   1. **NOTE**: this runs a fresh instance of the model; so the error values you see on the visualization may not align with those on the initial splash page.

   2. All visualizations will be stored in the 'visuals' folder. If you don't change the names or move the files yourself after creation,
similar problem-runs will overwrite the old ones.

   3. On interpretation: Circles are the hidden nodes, pluses are the training/test items. Size of hidden nodes codes for activation strength during that epoch. 
Color codes for category. When # classes = 2, the gradient of color between red and blue indicates how strongly a given node is associated with that category.
When classes = 3, hidden nodes are colored according to whichever category it is most strongly associated with (nominal, not continuous).

7. NOTE: This model is not presently set up to run gridsearches, perform regressions, or visualize problems with more than 3 classes.
