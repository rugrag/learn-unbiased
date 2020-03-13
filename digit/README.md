# Digit experiment
Download the digit dataset in form of .npy files from reference [1] and put them in the /data directory.

To run the code execute the digit_main.py file. Have a look at the digit_parser.py file to see the command line parameters.
Results in form of tensorboard log files will be stored in the /experiments folder.

Example of usage

# run baseline model
python3 digit_main.py --var '0.020' --lmb 0

# run our model
python3 digit_main.py --var '0.020' --lmb 1



you can try different values for lmb. The higher lmb, the more the representation is unbiased. Be careful that
for values of lmb too high, the model can not fit the training data, hence sub-optimal results.
