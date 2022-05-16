import numpy as np

def display_image(m, n, data):
    import tkinter as tk
    from PIL import Image, ImageTk

    imraw  = Image.new('1', (m, n))
    imraw.putdata([
        255 if data[i][j] else 0
        for j in range(n) for i in range(m)])

    root   = tk.Tk()
    image  = ImageTk.PhotoImage(image = imraw)
    canvas = tk.Canvas(root,width=300,height=300)

    canvas.pack()
    canvas.create_image(m, n, anchor="nw", image=image)

    root.mainloop()

def image_to_qtree(n, img):
    '''converts a matrix to a quadtree'''
    #basecase 1: if we have n=0: only 1 pixel
    #basecase 2: if we have plain img
    if n==0:
        return eval(f"('u', {img[0][0]})")
    else:
        black=0
        white=0
        for row in img:
            for r in row:
                if r==0:
                    black+=1
                if r==1:
                    white+=1
        if white==2**(2*n) and n!=0:
            return ('u', 1)
        if black==2**(2*n) and n!=0:
            return ('u', 0)
        '''print([img[i][:2 ** (n - 1)] for i in range(2**(n-1))])
        print([img[i][:2 ** (n - 1)] for i in range(2**(n-1),2**n)])
        print([img[i][2 ** (n - 1):] for i in range(2 ** (n - 1))])
        print([img[i][2 ** (n - 1):] for i in range(2 ** (n - 1), 2 ** n)])'''
        return eval(f"('c', ({image_to_qtree(n-1,[img[i][:2 ** (n - 1)] for i in range(2**(n-1))])}, {image_to_qtree(n-1,[img[i][2 ** (n - 1):] for i in range(2 ** (n - 1))])}, {image_to_qtree(n-1,[img[i][:2 ** (n - 1)] for i in range(2**(n-1),2**n)])}, {image_to_qtree(n-1,[img[i][2 ** (n - 1):] for i in range(2 ** (n - 1), 2 ** n)])}))")

print(image_to_qtree(1,[[1,0],
                        [1,1],
                        ]))
def concatenate(img, n, nw, ne, sw, se):

    for i in range(2**(n-1)):
        for j in range(2**(n-1)):
            img[i][j] = nw[1]
        for k in range(2**(n-1),2**n):
            img[i][k] = ne[1]
    for i in range(2**(n-1),2**n):
        for j in range(2 ** (n - 1)):
            img[i][j] = sw[1]
        for k in range(2 ** (n - 1), 2 ** n):
            img[i][k] = se[1]
    return img
node=(('u', 1), ('u', 0), ('u', 1), ('u', 1))
print(concatenate())
def qtree_to_image(n, node):
    '''convert from qtree back to matrix form for image'''

    if node[0] == 'u':
        print([[int(node[1])] * int(2**n)] * int(2**n))
        return [[int(node[1])] * int(2**n)] * int(2**n)
    else:
        print('pass')
        return qtree_to_image(n-1,node[1])

print(qtree_to_image(1,('c', (('u', 1), ('u', 0), ('u', 1), ('u', 1)))))
