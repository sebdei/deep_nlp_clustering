import csv


def provide_sequence_list(amount=-1):
    with open('data/Reviews.csv') as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        sentence_list = [row[9] for row in readCSV]

    return sentence_list[:amount]
