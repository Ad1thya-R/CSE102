import sys

def transacts_num(n):
    table=[0 for i in range(n+1)]
    table[0]=0
    coins=[1,3,7,9]
    c=len(coins)
    for i in range(1,n+1):
        for j in range(c):
            if coins[j]<=i:
                sub_res=table[i-coins[j]]
                if (sub_res!= sys.maxsize and sub_res+1<table[i]):


