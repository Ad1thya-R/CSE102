"""
The goal of this project is to develop a lossy image compression
format close to JPEG. Lossy compression algorithms allow to
greatly reduce the size of image files at the price of loosing some data from the original image. In general, a lossy image compression works by pruning the information the human eye is not sensible to. This is the case, for instance, in the JPEG file format.
"""
import math
import random as rd
import numpy as np
import scipy as sc
import time


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

def Cn(n: int) -> int:
	"""

	:param n:
	:return:
	"""
	C=[[0 for _ in range(n)] for _ in range(n)]
	for i in range(n):
		for j in range(n):
			if i == 0:
				C[i][j]=math.sqrt(1 / n)
			else:
				C[i][j] = math.sqrt(2 / n) * math.cos(math.pi / n * (j + 1 / 2) * i)
	return C

def Cn_T(n: int) -> list:
	"""

	:param n:
	:return:
	"""
	C = [[0 for _ in range(n)] for _ in range(n)]
	for i in range(n):
		for j in range(n):
			if i == 0:
				C[j][i] = math.sqrt(1 / n)
			else:
				C[j][i] = math.sqrt(2 / n) * math.cos(math.pi / n * (j + 1 / 2) * i)
	return C


def DCT(v: list) -> list:
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

def IDCT(v: list) -> list :
	"""

	:param v:
	:return:
	"""
	n=len(v)
	C=Cn(n)
	result = []
	for i in range(len(C[0])):  # this loops through columns of the matrix
		total = 0
		for j in range(len(v)):  # this loops through vector coordinates & rows of matrix
			total += v[j] * C[j][i]
		result.append(total)
	return result


def matProd (A , B ):
	# we assume that A and B are non - empty matrices
	m = len ( A )
	n = len ( A [0])
	if len ( B ) != n :
		return None # the size does not match
	p = len ( B [0])
	C = [ None ] * m
	for i in range ( m ):
		C [ i ] = [0] * p
		for k in range ( p ):
			for j in range ( n ):
				C[i][k] += A[i][j] * B[j][k]
	return C

def DCT2(m: int, n: int, A: list):
	"""

	:param m: height (no of rows)
	:param n: width (no of columns)
	:param A: input matrix
	:return:
	"""
	CnT=Cn_T(n)
	Cm=Cn(m)
	first_mult=matProd(A,CnT)
	return matProd(Cm,first_mult)

def IDCT2(m: int, n: int, A: list):
	"""

	:param m: height (no of rows)
	:param n: width (no of columns)
	:param A: input matrix
	:return:
	"""
	C = Cn(n)
	Cm = Cn_T(m)
	first_mult = matProd(A, C)
	return matProd(Cm, first_mult)

def redalpha(i: int) -> tuple:
	"""

	:param i:
	:return:
	s, an integer in the set {−1,1},
	k an integer in the range {0..8},
	αi=s⋅αk
	"""
	if abs(i)<=8:
		return 1,abs(i)
	else:
		int_div=i//16
		remainder=i%16
		if int_div%2==0:
			k=1
		else:
			k=-1
		if remainder>=9:
			remainder-=16
			k*=-1
	return (k,abs(remainder))
print(redalpha(-8))
def ncoeff8(i, j):
	"""

	:param i: integer in the range {0..8},
	:param j: integer in the range {0..8},
	:return:
	s, an integer in the set {−1,1},
	k an integer in the range {0..7},
	Cij=s⋅αk
	"""
	if i == 0:
		return redalpha(4)
	else:
		return redalpha(i*(2*j+1))

#pre-compute C_alpha matrix
C_alpha=[
    [1/2*ncoeff8(i, j)[0]*math.cos(ncoeff8(i,j)[1]*math.pi/16) for j in range(8)]
    for i in range(8)
]
v_before=[[0 for _ in range(8)] for _ in range(8)]
v_final = [[0 for _ in range(8)] for _ in range(8)]
print(C_alpha)
print(Cn(8))
def DCT_Chen(A):
	"""

	:param A:
	:return:
	58
	"""

	for r in range(8):
		v0=C_alpha[4][0] * sum([A[r][j] for j in range(8)])
		v_before[r][0]=v0
		for i in range(1,8):
			if i%2==0:
				if i==4:
					v_add=(A[r][0]+ A[r][7])+(A[r][3]+ A[r][4])-(A[r][1]+ A[r][6])-(A[r][2]+ A[r][5])
					v_temp=C_alpha[4][0]*v_add
				elif i==6:
					v_add1 = (A[r][0] + A[r][7]) - (A[r][3] + A[r][4])
					v_add2 = (A[r][1] + A[r][6]) - (A[r][2] + A[r][5])
					v_temp=C_alpha[6][0]*v_add1-C_alpha[2][0]*v_add2
				else:
					v_temp=sum([(A[r][j]+ A[r][7-j]) * C_alpha[i][j] for j in range(4)])
			else:
				if i==3:
					v_add1=(A[r][0]+ A[r][7])-(A[r][3]+ A[r][4])
					v_add2=(A[r][1]+ A[r][6])-(A[r][2]+ A[r][5])
					v_temp=C_alpha[2][0]*v_add1+C_alpha[6][0]*v_add2
				else:
					v_temp = sum([(A[r][j] - A[r][7 - j]) * C_alpha[i][j] for j in range(4)])
			v_before[r][i]=v_temp
	v_mid=[[v_before[j][i] for j in range(len(v_before))] for i in range(len(v_before[0]))]
	for r in range(8):
		v0 = math.cos(math.pi / 4) / 2 * sum([v_mid[r][j] for j in range(8)])
		v_final[0][r] = v0
		for i in range(1, 8):
			if i % 2 == 0:
				if i == 4:
					v_add = (v_mid[r][0] + v_mid[r][7]) + (v_mid[r][3] + v_mid[r][4]) - (v_mid[r][1] + v_mid[r][6]) - (v_mid[r][2] + v_mid[r][5])
					v_temp = C_alpha[4][0] * v_add
				elif i == 6:
					v_add1 = (v_mid[r][0] + v_mid[r][7]) - (v_mid[r][3] + v_mid[r][4])
					v_add2 = (v_mid[r][1] + v_mid[r][6]) - (v_mid[r][2] + v_mid[r][5])
					v_temp = C_alpha[6][0] * v_add1 - C_alpha[2][0] * v_add2
				else:
					v_temp = sum([(v_mid[r][j] + v_mid[r][7 - j]) * C_alpha[i][j] for j in range(4)])
			else:
				if i == 3:
					v_add1 = (v_mid[r][0] + v_mid[r][7]) - (v_mid[r][3] + v_mid[r][4])
					v_add2 = (v_mid[r][1] + v_mid[r][6]) - (v_mid[r][2] + v_mid[r][5])
					v_temp = C_alpha[2][0] * v_add1 + C_alpha[6][0] * v_add2
				else:
					v_temp = sum([(v_mid[r][j] - v_mid[r][7 - j]) * C_alpha[i][j] for j in range(4)])
			v_final[i][r] = v_temp
	return v_final


def IDCT_Chen(A):
	"""

	:param A:
	:return:
	"""






M8 = [
    [ncoeff8(i, j) for j in range(8)]
    for i in range(8)
]


def M8_to_str(M8):
	def for1(s, i):
		return f"{'+' if s >= 0 else '-'}{i:d}"


	return "\n".join(
		" ".join(for1(s, i) for (s, i) in row)
		for row in M8
	)

print(M8_to_str(M8))

dctmat=[
  [140,  144,  147,  140,  140,  155,  179,  175],
  [144,  152,  140,  147,  140,  148,  167,  179],
  [ 152,  155,  136,  167,  163,  162,  152,  172],
  [168,  145,  156,  160,  152,  155,  136,  160],
  [162,  148,  156,  148,  140,  136,  147,  162],
  [ 147,  167,  140,  155,  155,  140,  136,  162],
  [ 136,  156,  123,  167,  162,  144,  140,  147],
  [ 148,  155,  136,  155,  152,  147,  147,  136]
]
start = time.process_time()

for _ in range(20):
	print('nai:', DCT2(8,8,dctmat))

	end = time.process_time()

	print(end - start)


	start2 = time.process_time()


	print('opt:', DCT_Chen(dctmat))

	end2 = time.process_time()

	print(end2 - start2)


v=[8, 16, 24, 32, 40, 48, 56, 64]
v2=[101.82, -51.54, -0.0, -5.39, 0.0, -1.61, -0.0, -0.41]
print(sum([1 / math.sqrt(len(v2)) * v2[k] for k in range(len(v2))]))
print(IDCT(v2))




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




mat_sampled=[[2, 3, 4, 2],
			 [2, 3, 4, 2],
			 [1, 2, 2, 1]]




