"""
The goal of this project is to develop a lossy image compression
format close to JPEG. Lossy compression algorithms allow to
greatly reduce the size of image files at the price of loosing some data from the original image. In general, a lossy image compression works by pruning the information the human eye is not sensible to. This is the case, for instance, in the JPEG file format.
"""
import math

#The following class operates to check the validity of User inputs
class InvalidInput(Exception):
	def __init__(self, data, correct):
		super().__init__()
		self.data = data
		self.correct = correct
	def __str__(self):
		return f'The input {self.data} is invalid. The input should be of the following form: {self.correct}'

#Start of PROJECT
def ppm_tokenize(stream):
	"""
	:param stream: input stream such as g=open('file')
	:return: iterator (generator) that outputs the data individually as tokens
	"""
	for line in stream:
		line = line.partition('#')[0]
		line = line.rstrip().split()
		if line:
			for t in line:
				yield t



def ppm_load(stream):
	"""
	:param stream: input stream such as g=open('file')
	:return: width, height and 2D-array for image
	"""
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
	return int(w),int(h),img

def ppm_save(w: int, h: int, img: list, output) -> None:
	"""takes an output stream output and
	that saves the PPM image img whose size is w x h."""
	output.write('P3\n')
	output.write(f'{w} {h}\n')
	output.write('255\n')
	for row in img:
		for pix in row:
			output.write(f'{pix}\n')



def RGB2YCbCr(r: int, g: int, b: int) -> tuple:
	"""
	:int r: red, 0 to 255
	:int g: green 0 to 255
	:int b: blue 0 to 255
	:return: Y, Cb, Cr color format
	"""
	Y = 0.299 * r + 0.587 * g + 0.114 * b
	Cb = 128 - 0.168736 * r - 0.331264 * g + 0.5 * b
	Cr = 128 + 0.5 * r - 0.418688 * g - 0.081312 * b
	return (round(Y), round(Cb), round(Cr))



def YCbCr2RGB(Y: int, Cb: int, Cr: int) -> tuple:
	"""
	:int r: red, 0 to 255
	:int g: green 0 to 255
	:int b: blue 0 to 255
	:return: r,g,b color format as integers rounded
	"""
	r = Y + 1.402 * (Cr - 128)
	g = Y - 0.34414 * (Cb - 128) - 0.71414 * (Cr - 128)
	b = Y + 1.772 * (Cb - 128)
	return (round(r), round(g), round(b))



def img_RGB2YCbCr(img: list) -> list:
	"""
	:param img: 2D array with pixels stored as rgb values ('255 255 255')
	:return: Y[i][j], Cb[i][j], Cr[i][j] --> 3 matrices with these attributes
	and i,j denoting the position of the pixel
	"""
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

def img_YCbCr2RGB(Y: int, Cb: int, Cr: int) -> list:
	"""

	:param Y: 2D array of values of Y for each pixel in image
	:param Cb: 2D array of values of Cb for each pixel in image
	:param Cr: 2D array of values of Cr for each pixel in image
	:return: converts img back to pixels with RGB formatting from YCbCr
	"""
	img=[]
	for i in range(len(Y)):
		row=[]
		for j in range(len(Y[0])):
			r,g,b=YCbCr2RGB(Y[i][j], Cb[i][j], Cr[i][j])
			row.append(str(r)+' '+str(g)+' '+str(b))
		img.append(row)
	return img

def subsampling(w : int, h: int, C: list, b: int, a: int) -> list:
	"""
	:param w: width of image
	:param h: height of image
	:param C: 2D array of certain attribute (e.g. RGB or Y attributes)
	:param b: sub samplng mode dividing the image's width (block height)
	:param a:sub sampling mode dividing the image's height (block width)
	:return: an img where the data has been compressed by considering
	axb blocks as pixels and averaging out the values in these blocks
	"""
	#Check if input C is matrix
	for row in C:
		if len(C[0])!=len(row):
			raise InvalidInput(C, "A matrix of integers (A 2D array where the rows are of equal length).")
	#Check if dimensions match input array
	if h!=len(C) or w!=len(C[0]):
		raise InvalidInput(f'{(w,h)} are not the dimensions of the input matrix, it is', f'please input the correct dimensions of the matrix img, {len(C)}x{len(C[0])}')
	#inputs a and b cannot be bigger than the dimensions of the matrix
	if a>w or b>h:
		raise InvalidInput((a,b),"integers a and b s.t. a<=w and b<=h")

	#Code for the function
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


def extrapolate(w: int, h: int, C: list, b: int, a: int) -> list:
	"""
	:param w: width BEFORE sub sampling has been applied
	:param h: height BEFORE sub sampling has been applied
	:param C: sub sampled matrix
	:param b:sub samplng mode dividing the image's width (block height)
	:param a:sub sampling mode dividing the image's height (block width)
	:return: A 2D-array to emulate the original image (as some data was lost during subsampling)
	"""

	##Testing whether input is correct

	##Check whether 2D array is a matrix
	for row in C:
		if len(C[0])!=len(row):
			raise InvalidInput(C, "A matrix of integers (A 2D array where the rows are of equal length).")

	img=[]
	for smallrow in C:
		temp_row=[]
		for pixel in smallrow:
			temp_row+=[pixel]*a
			temp_row=temp_row[:w]
		img.extend([temp_row]*b)
	img=img[:h]
	return img


def block_splitting(w: int, h: int, C: list) -> list:
	"""

	:param w:
	:param h:
	:param C:
	:return:
	"""
	for i in range(0, h, 8):
		block = []
		if i + 8 > h:
			temp_col = C[i:i + 8]
			t = len(temp_col)
			temp_col.extend([temp_col[t - 1]] * (8 - t))
			C_cut=temp_col
		else:
			C_cut=C[i:i+8]
		for j in range(0, w, 8):
			for c in C_cut:
				if j + 8 > w:
					temp_row=c[j:j+8]
					t=len(temp_row)
					temp_row.extend([temp_row[t-1]]*(8-t))
					block.append(temp_row)
				else:
					block.append(c[j:j+8])
		if len(block)==16:
			yield block[:8]

			yield block[8:]
		else:
			yield block

def DCT(v):
	"""

	:param v:
	:return:
	"""
	n=len(v)
	v_1=sum([1/math.sqrt(n)*v[j] for j in range(n)])
	v_out=[round(v_1,2)]
	for i in range(1,n):
		temp_v=sum([math.sqrt(2/n) * v[j] * math.cos(math.pi/n * (j + 1/2) * i) for j in range(n)])
		v_out.append(round(temp_v,2))
	return v_out

def IDCT(v):
	"""

	:param v:
	:return:
	"""



v=[8, 16, 24, 32, 40, 48, 56, 64]
print(DCT(v))




C=[
	[ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10],
    [ 2,  3,  4,  5,  6,  7,  8,  9, 10,  1],
    [ 3,  4,  5,  6,  7,  8,  9, 10,  1,  2],
    [ 4,  5,  6,  7,  8,  9, 10,  1,  2,  3],
    [ 5,  6,  7,  8,  9, 10,  1,  2,  3,  4],
    [ 6,  7,  8,  9, 10,  1,  2,  3,  4,  5],
    [ 7,  8,  9, 10,  1,  2,  3,  4,  5,  6],
    [ 8,  9, 10,  1,  2,  3,  4,  5,  6,  7],
    [ 9, 10,  1,  2,  3,  4,  5,  6,  7,  8],
]

a=block_splitting(10,9,C)

print(next(a))
print(next(a))
print(next(a))
print(next(a))


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


print(subsampling(6,10,mat,4,3))

mat_sampled=[[2, 3, 4, 2],
			 [2, 3, 4, 2],
			 [1, 2, 2, 1]]

print(extrapolate(10,10,mat_sampled,4,3))

