#!/bin/sh
# Use this to import the twitter training data into MongoDB

mongoimport --db tweets --collection training --type csv --fields polarity,_id,date,query,user,text --file training.1600000.processed.noemticon.csv
