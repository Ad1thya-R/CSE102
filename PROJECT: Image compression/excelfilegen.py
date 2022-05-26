import pandas as pd
import csv

new_list = [
    [4,4,4,4,4,4,4,4],
	[2,6,-6,-2,2,6,-6,-2],
	[4,-4,-4,4,4,-4,-4,4],
	[6,-2,2,-6,6,-2,2,-6],
	[1,3,5,7,-1,-3,-5,-7],
	[3,-7,-1,-5,-3,7,1,5],
	[5,-1,7,3,-5,1,-7,-3],
	[7,-5,3,-1,-7,5,-3,1]
]

with open('GFG.csv', 'w') as f:
	# using csv.writer method from CSV package
	write = csv.writer(f)
	write.writerows(new_list)