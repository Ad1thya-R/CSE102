def huffman_stats(s):
    '''takes a string s and returns a dictionary
     whose keys are the characters c appearing in s
    and whose associated value p is the appearance ratio of c in s'''
    stats={}
    for l in s:
        if l in stats.keys():
            stats[l]+=1/len(s)
        else:
            stats[l]=1/len(s)
    return stats

def huffman_tree(d):
    '''takes a dictionary d associating weights
    to characters (e.g., a dictionary as
    returned by the previous function huffman_stats(s)),
    constructs an Huffman tree for d and returns its root.'''
    sort_d={k: v for k, v in sorted(d.items(), key=lambda item: item[1])}
    for k,v in sort_d.items():
        pass



