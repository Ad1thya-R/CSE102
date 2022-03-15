def slice_dice(n, s, dice):
    return [dice[i * s:i * s + s] for i in range(n)]
def win_probability(die1,die2):
    counter=0
    for one in die1:
        for two in die2:
            if one>two:
                counter+=1
    return counter/(len(die1)*len(die2))

def beats(die1,die2):
    return True if win_probability(die1,die2)>0.5 else False

def get_dice(n,s,dice):
    die=slice_dice(n,s,dice)
    if n*s==len(dice):
        if beats(die[n-1],die[0]):
            yield dice
        else:
            return
    else:
        r=len(dice)//2



