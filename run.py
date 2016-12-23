#!usr/bin/python

import sys
sys.path.insert(0, './src')
import lstm
import argparse
import json


def init_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", help="input config JSON file")
    parser.add_argument("-i", "--infile", help="input file (csv)")
    parser.add_argument("-o", "--outfile", help="output file")
    parser.add_argument("-lf", "--logfile", help="log file to store the training time & RMSE value")
    parser.add_argument("-nl", "--nlayers", help="number of layers", type=int)
    parser.add_argument("-dfu", "--dropout_fraction_ru", help="fraction of R_Units to be dropped [0.0, 1.0)", type=float)
    parser.add_argument("-dfw", "--dropout_fraction_rw", help="fraction of input units to be dropped [0.0, 1.0)", type=float)
    parser.add_argument("-ld", "--layerdim", help="layer dimensions", nargs="+", type=int)
    parser.add_argument("-opt", "--optimizer", help="name of the optimizer")
    parser.add_argument("-lr", "--learnrate", help="the learning rate", type=float)
    parser.add_argument("-m", "--momentum", help="the momentum", type=float)
    parser.add_argument("-trp", "--trainpct", help="training percent (0.0, 1.0]", type=float)
    parser.add_argument("-em", "--errmetric", help="type of error metric")
    parser.add_argument("-a", "--append", help="append run configuration to logfile", type=bool, default=False, const=True, nargs="?")
    parser.add_argument("-e", "--epoch", help="number of epochs to run", type=int)
    return parser

if __name__ == "__main__":
    parser = init_args()
    args = parser.parse_args()
    given = {
        "input_filename": args.infile,
        "output_filename": args.outfile,
        "n_layers": 3,
        "dropout_fraction_ru": 0,
	    "dropout_fraction_rw": 0,
        "layer_dimensions": [1,4,1],
        "optimizer": "adam",
        "learning_rate": 0.01,
        "momentum": 0.0,
        "training_percent": 0.7,
        "err_metric": "mean_squared_error",
        "logfile": None,
	    "epoch": 100,
    }
    # Checking for required fields
    if args.config:
    	if not args.infile or not args.outfile:
    	    print "Please provide inputfile and outputfile."
    	    parser.print_help()
        else:
    	    jsonDict = {}
    	    with open(args.config) as dataJSON:
    	       jsonDict = dict(json.loads(dataJSON.read()))
            for k, v in jsonDict.iteritems():
    	       given[k] = v
    	    # set configuration
    	    lstm.set_configuration(input_filename=given["input_filename"],
    		output_filename=given["output_filename"], n_layers=given["n_layers"], dropout_fraction_ru=given["dropout_fraction_ru"], dropout_fraction_rw=given["dropout_fraction_rw"],
    		layer_dimensions=given["layer_dimensions"], optimizer=given["optimizer"], learning_rate=given["learning_rate"], momentum=given["momentum"], 
    		training_percent=given["training_percent"], err_metric=given["err_metric"], logfile=given["logfile"], epoch=given["epoch"])
    	    # run
    	    lstm.run()
    	    # checking whether to append run configuration to run_configs file
    	    if args.append:
    		  lstm.append_config()
    else:
        if not args.infile or not args.outfile or not args.learnrate:
            parser.print_help()
        else:
            if args.nlayers:
                given["n_layers"] = args.nlayers
            if args.dropout_fraction_ru:
                given["dropout_fraction_ru"] = args.dropout_fraction_ru
	        if args.dropout_fraction_rw:
		        given["dropout_fraction_rw"] = args.dropout_fraction_rw
            if args.layerdim:
                given["layer_dimensions"] = args.layerdim
            if args.optimizer:
                given["optimizer"] = args.optimizer
            if args.learnrate:
                given["learning_rate"] = args.learnrate
            if args.momentum:
                given["momentum"] = args.momentum
            if args.trainpct:
                given["training_percent"] = args.trainpct
            if args.errmetric:
                given["err_metric"] = args.errmetric
	        if args.logfile:
	            given["logfile"] = args.logfile
	        if args.epoch:
		        given["epoch"] = args.epoch
    	    # set configuration
    	    lstm.set_configuration(input_filename=given["input_filename"],
    		output_filename=given["output_filename"], n_layers=given["n_layers"], dropout_fraction_ru=given["dropout_fraction_ru"], dropout_fraction_rw=given["dropout_fraction_rw"],
    		layer_dimensions=given["layer_dimensions"], optimizer=given["optimizer"], learning_rate=given["learning_rate"], momentum=given["momentum"], 
    		training_percent=given["training_percent"], err_metric=given["err_metric"], logfile=given["logfile"], epoch=given["epoch"])
    	    # run
    	    lstm.run()
    	    # checking whether to append run configuration to run_configs file
    	    if args.append:
    		  lstm.append_config()

