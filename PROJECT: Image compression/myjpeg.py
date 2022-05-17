from itertools import repeat
'''
The goal of this project is to develop a lossy image compression
format close to JPEG. Lossy compression algorithms allow to
greatly reduce the size of image files at the price of loosing some data from the original image. In general, a lossy image compression works by pruning the information the human eye is not sensible to. This is the case, for instance, in the JPEG file format.
'''


def ppm_tokenize(stream):
	'''

	:param stream: input stream such as g=open('file')
	:return: iterator (generator) that outputs the data individually as tokens
	'''
	for line in stream:
		line = line.partition('#')[0]
		line = line.rstrip().split()
		if line:
			for t in line:
				yield t


def ppm_load(stream):
	'''
	:param stream: input stream such as g=open('file')
	:return: width, height and 2D-array for image
	'''
	g=ppm_tokenize(stream)
	img=[]
	for i in range(4):
		curr=next(g)
		if i==1:
			w = curr
		if i==2:
			h = curr

	for i in range(int(h)):
		temprow=[]
		for j in range(int(w)):
			curr2=[next(g) for _ in range(3)]
			temprow.append(' '.join(curr2))
		img.append(temprow)
		temprow=[]
	return int(w),int(h),img

def ppm_save(w,h,img, output):
	'''takes an output stream output and
	that saves the PPM image img whose size is w x h.'''
	output.write('P3\n')
	output.write(f'{w} {h}\n')
	output.write('255\n')
	for row in img:
		for pix in row:
			output.write(f'{pix}\n')

outf=open('writefile', 'w')

inp=open('file')
print(ppm_load(inp))

ppm_save(3, 2, [['0 255 0', '0 0 255', '255 255 0'], ['255 255 255', '0 0 0', '0 0 0']],outf)

def RGB2YCbCr(r, g, b):
	'''

	:int r: red, 0 to 255
	:int g: green 0 to 255
	:int b: blue 0 to 255
	:return:
	'''
	Y = 0.299 * r + 0.587 * g + 0.114 * b
	Cb = 128 - 0.168736 * r - 0.331264 * g + 0.5 * b
	Cr = 128 + 0.5 * r - 0.418688 * g - 0.081312 * b
	return (Y, Cb, Cr)


def YCbCr2RGB(Y, Cb, Cr):
	'''
	:int r: red, 0 to 255
	:int g: green 0 to 255
	:int b: blue 0 to 255
	:return:
	'''
	r = Y + 1.402 * (Cr - 128)
	g = Y - 0.34414 * (Cb - 128) - 0.71414 * (Cr - 128)
	b = Y + 1.772 * (Cb - 128)
	return (r, g, b)



def img_RGB2YCbCr(img):
	'''
	:param img: 2D array with pixels stored as rgb values ('255 255 255')
	:return: Y[i][j], Cb[i][j], Cr[i][j] --> 3 matrices with these attributes
	and i,j denoting the position of the pixel
	'''
	h=len(img)
	w=len(img[0])
	Y=[]
	Cb=[]
	Cr=[]
	for i, row in enumerate(img):
		Y_row=[]
		Cb_row = []
		Cr_row = []
		for j, pixel in enumerate(row):
			(R,G,B)=tuple(pixel.split())
			Y_pixel, Cb_pixel, Cr_pixel = RGB2YCbCr(float(R),float(G),float(B))
			Y_row.append(Y_pixel)
			Cb_row.append(Cb_pixel)
			Cr_row.append(Cr_pixel)
		Y.append(Y_row)
		Cb.append(Cb_row)
		Cr.append(Cr_row)
	return (Y, Cb, Cr)

def img_YCbCr2RGB(Y, Cb, Cr):
	'''

	:param Y: 2D array of values of Y for each pixel in image
	:param Cb: 2D array of values of Cb for each pixel in image
	:param Cr: 2D array of values of Cr for each pixel in image
	:return: converts img back to pixels with RGB formatting from YCbCr
	'''
	img=[]
	for i in range(len(Y)):
		row=[]
		for j in range(len(Y[0])):
			r,g,b=YCbCr2RGB(Y[i][j], Cb[i][j], Cr[i][j])
			row.append(str(r)+' '+str(g)+' '+str(b))
		img.append(row)
	return img

def subsampling(w,h,C,b,a):
	'''
	:param w: width of image
	:param h: height of image
	:param C: 2D array of certain attribute (e.g. RGB or Y attributes)
	:param b: sub samplng mode dividing the image's width (block height)
	:param a:sub sampling mode dividing the image's height (block width)
	:return:
	'''
	img = []
	for i in range(0, h, b):
		row = []
		C_cut=C[i:i+b]
		for j in range(0, w, a):
			temp_sum=0
			for c in C_cut:
				temp_sum+=sum(c[j:j+a])
				temp_len=len(c[j:j+a])
			row.append(round(temp_sum/(temp_len*len(C_cut))))
		img.append(row)
	return img

def extrapolate(w, h, C, b, a):
	'''
	:param w: width BEFORE sub sampling has been applied
	:param h: height BEFORE sub sampling has been applied
	:param C: sub sampled matrix
	:param b:sub samplng mode dividing the image's width (block height)
	:param a:sub sampling mode dividing the image's height (block width)
	:return:
	'''
	img=[]
	for smallrow in C:
		column_count=0
		temp_row=[]
		for pixel in smallrow:
			temp_row+=[pixel]*a
			temp_row=temp_row[:w]
		img.extend([temp_row]*b)
	img=img[:h]
	return img



print([[2]*3+[3]*3+[4]*4]*10)

mat=[[2, 2, 2, 3, 3, 3, 4, 4, 4, 5],
	 [2, 2, 2, 3, 3, 3, 4, 4, 4, 6],
	 [2, 2, 2, 3, 3, 3, 4, 4, 4, 7],
	 [2, 2, 2, 3, 3, 3, 4, 4, 4, 1],
	 [2, 2, 2, 3, 3, 3, 4, 4, 4, 2],
	 [2, 2, 2, 3, 3, 3, 4, 4, 4, 3],
	 [2, 2, 2, 3, 3, 3, 4, 4, 4, 6],
	 [2, 2, 2, 3, 3, 3, 4, 4, 4, 7],
	 [2, 2, 2, 3, 3, 3, 4, 4, 4, 8],
	 [2, 2, 2, 3, 3, 3, 4, 4, 4, 9]]



print(img_RGB2YCbCr([['0 255 0', '0 0 255', '255 255 0'], ['255 255 255', '0 0 0', '0 0 0']]))
print(img_YCbCr2RGB([[149.685, 29.07, 225.93], [255.0, 0.0, 0.0]], [[43.527680000000004, 255.5, 0.5], [128.0, 128.0, 128.0]], [[21.234560000000002, 107.26544, 148.73456], [127.99999999999999, 128.0, 128.0]]))


print(subsampling(10,10,mat,4,3))

mat_sampled=[[2, 3, 4, 4],
			 [2, 3, 4, 4],
			 [2, 3, 4, 8]]

print(extrapolate(10,10,mat_sampled,4,3))