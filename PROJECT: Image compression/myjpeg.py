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

def Transpose(A):
	"""

	:param A: matrix (in form of list of lists)
	:return: Transpose of matrix A
	"""
	return [[A[j][i] for j in range(len(A))] for i in range(len(A[0]))]

#pre-compute C_alpha matrix
C_alpha=[
    [1/2*ncoeff8(i, j)[0]*math.cos(ncoeff8(i,j)[1]*math.pi/16) for j in range(8)]
    for i in range(8)
]
C_coeff=[]
v_before=[[0 for _ in range(8)] for _ in range(8)]
v_final = [[0 for _ in range(8)] for _ in range(8)]

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
				elif i==2:
					v_add1 = (A[r][0] + A[r][7]) - (A[r][3] + A[r][4])
					v_add2 = (A[r][1] + A[r][6]) - (A[r][2] + A[r][5])
					v_temp = C_alpha[2][0] * v_add1 + C_alpha[6][0] * v_add2
				elif i==6:
					v_add1 = (A[r][0] + A[r][7]) - (A[r][3] + A[r][4])
					v_add2 = (A[r][1] + A[r][6]) - (A[r][2] + A[r][5])
					v_temp=C_alpha[6][0]*v_add1-C_alpha[2][0]*v_add2
				else:
					v_temp=sum([(A[r][j]+ A[r][7-j]) * C_alpha[i][j] for j in range(4)])
			else:
				v_temp = sum([(A[r][j] - A[r][7 - j]) * C_alpha[i][j] for j in range(4)])
			v_before[r][i]=v_temp
	v_mid=Transpose(v_before)
	for r in range(8):
		v0 = math.cos(math.pi / 4) / 2 * sum([v_mid[r][j] for j in range(8)])
		v_final[0][r] = v0
		for i in range(1, 8):
			if i % 2 == 0:
				if i == 4:
					v_add = (v_mid[r][0] + v_mid[r][7]) + (v_mid[r][3] + v_mid[r][4]) - (v_mid[r][1] + v_mid[r][6]) - (v_mid[r][2] + v_mid[r][5])
					v_temp = C_alpha[4][0] * v_add
				elif i == 2:
					v_add1 = (v_mid[r][0] + v_mid[r][7]) - (v_mid[r][3] + v_mid[r][4])
					v_add2 = (v_mid[r][1] + v_mid[r][6]) - (v_mid[r][2] + v_mid[r][5])
					v_temp = C_alpha[2][0] * v_add1 + C_alpha[6][0] * v_add2
				elif i == 6:
					v_add1 = (v_mid[r][0] + v_mid[r][7]) - (v_mid[r][3] + v_mid[r][4])
					v_add2 = (v_mid[r][1] + v_mid[r][6]) - (v_mid[r][2] + v_mid[r][5])
					v_temp = C_alpha[6][0] * v_add1 - C_alpha[2][0] * v_add2
				else:
					v_temp = sum([(v_mid[r][j] + v_mid[r][7 - j]) * C_alpha[i][j] for j in range(4)])
			else:
				v_temp = sum([(v_mid[r][j] - v_mid[r][7 - j]) * C_alpha[i][j] for j in range(4)])
			v_final[i][r] = v_temp
	return v_final

##Output comparison for DCT
naive=[[1210.0, -17.99692735337386, 14.779256233067407, -8.979557300103124, 23.250000000000014, -9.232556779784952, -13.969111825644514, -18.937081397077478],
	   [20.53822758002366, -34.0929440302768, 26.330398745736908, -9.038521236678951, -10.93299899504348, 10.730687995887013, 13.772435090993131, 6.9547727586290815],
	   [-10.38402933509522, -23.514074296063256, -1.853553390593277, 6.0404645982005585, -18.07450986222912, 3.1969718830684295, -20.41726188957802, -0.8264830874559816],
	   [-8.104806529195912, -5.04145665135767, 14.332151475584137, -14.613395187278638, -8.217831534021615, -2.732392543690959, -3.084509970003599, 8.429155764630426],
	   [-3.249999999999943, 9.501426153921102, 7.88463853042961, 1.317025177983379, -10.999999999999991, 17.904478829585752, 18.381919792134475, 15.24116083384256],
	   [3.8556330205585283, -2.2146896705631494, -18.166970222878785, 8.499840503342673, 8.268830646150814, -3.608430219669282, 0.8689979628581561, -6.862524463223144],
	   [8.901372633918172, 0.6330199687857647, -2.91726188957802, 3.6413659899039135, -1.1724304841575934, -7.421804998309854, -1.146446609406703, -1.9245633104352884],
	   [0.04912235980940949, -7.812994194337432, -2.4245087494156357, 1.5903798159784732, 1.1992571025829004, 4.247012669253736, -6.417410588884525, 0.31476943722475337]]

chen=[[1210.0000000000002, -17.996927353373888, 14.77925623306755, -8.97955730010327, 23.250000000000004, -9.232556779784863, -13.969111825643534, -18.937081397078945],
	  [20.538227580023698, -34.09294403027681, 26.330398745736915, -9.038521236678974, -10.932998995043473, 10.730687995887013, 13.772435090993197, 6.954772758628977],
	  [-10.384029335095121, -23.51407429606327, -1.8535533905932715, 6.040464598200588, -18.07450986222913, 3.196971883068438, -20.417261889578036, -0.8264830874559868],
	  [-8.104806529196079, -5.041456651357671, 14.332151475584148, -14.613395187278673, -8.217831534021599, -2.732392543690965, -3.0845099700035745, 8.429155764630408],
	  [-3.2499999999999862, 9.501426153921093, 7.884638530429614, 1.317025177983375, -11.000000000000002, 17.90447882958578, 18.381919792134465, 15.241160833842573],
	  [3.855633020558628, -2.2146896705631507, -18.16697022287876, 8.499840503342668, 8.268830646150828, -3.608430219669304, 0.8689979628581361, -6.862524463223146],
	  [8.901372633919138, 0.6330199687857636, -2.9172618895780342, 3.6413659899039232, -1.172430484157588, -7.421804998309837, -1.146446609406727, -1.9245633104353126],
	  [0.0491223598079511, -7.81299419433739, -2.424508749415626, 1.5903798159784621, 1.199257102582877, 4.247012669253758, -6.417410588884475, 0.31476943722478534]]
############################################################################
C_alpha_prov=[
    [4,4,4,4,4,4,4,4],
	[2,6,-6,-2,2,6,-6,-2],
	[4,-4,-4,4,4,-4,-4,4],
	[6,-2,2,-6,6,-2,2,-6],
	[1,3,5,7,-1,-3,-5,-7],
	[3,-7,-1,-5,-3,7,1,5],
	[5,-1,7,3,-5,1,-7,-3],
	[7,-5,3,-1,-7,5,-3,1]
]

[[0.3535533905932738, 0.3535533905932738, 0.3535533905932738, 0.3535533905932738, 0.3535533905932738, 0.3535533905932738, 0.3535533905932738, 0.3535533905932738],
 [0.46193976625564337, 0.19134171618254492, -0.19134171618254492, -0.46193976625564337, 0.46193976625564337, 0.19134171618254492, -0.19134171618254492, -0.46193976625564337],
 [0.3535533905932738, -0.3535533905932738, -0.3535533905932738, 0.3535533905932738, 0.3535533905932738, -0.3535533905932738, -0.3535533905932738, 0.3535533905932738],
 [0.19134171618254492, -0.46193976625564337, 0.46193976625564337, -0.19134171618254492, 0.19134171618254492, -0.46193976625564337, 0.46193976625564337, -0.19134171618254492],
 [0.4903926402016152, 0.4157348061512726, 0.27778511650980114, 0.09754516100806417, -0.4903926402016152, -0.4157348061512726, -0.27778511650980114, -0.09754516100806417],
 [0.4157348061512726, -0.09754516100806417, -0.4903926402016152, -0.27778511650980114, -0.4157348061512726, 0.09754516100806417, 0.4903926402016152, 0.27778511650980114],
 [0.27778511650980114, -0.4903926402016152, 0.09754516100806417, 0.4157348061512726, -0.27778511650980114, 0.4903926402016152, -0.09754516100806417, -0.4157348061512726],
 [0.09754516100806417, -0.27778511650980114, 0.4157348061512726, -0.4903926402016152, -0.09754516100806417, 0.27778511650980114, -0.4157348061512726, 0.4903926402016152]]

##pre-compute C_alpha2 matrix
C_alpha2=[
    [1/2*ncoeff8(i, j)[0]*math.cos(ncoeff8(i,j)[1]*math.pi/16) for j in range(8)]
    for i in range(8)
]
C_coeff=[[ncoeff8(i,j)[0]*ncoeff8(i,j)[1] for j in range(8)] for i in range(8)]
v_before2=[0 for _ in range(8)]
v_final2 = [[0 for _ in range(8)] for _ in range(8)]
print(C_alpha2)
'''
def IDCT_Chen(A):
	"""

	:param A:
	:return: Matrix which reverses the process of Discrete Cosine Transform
	"""

	def I_1DCT_Chen(v_hat):
		"""
		1D implementation of Chen algorithm
		for multiplicative optimisation of IDCT_Chen
		:param A:
		:return:
		"""
		cache={(0,4): v_hat[0]*C_alpha2[0][0],
			   (2,2): v_hat[2]*C_alpha2[1][0],
			   (2,6): v_hat[2]*C_alpha2[1][1],
			   (4,4): v_hat[4]*C_alpha2[2][0],
			   (6,2): v_hat[6]*C_alpha2[1][0],
			   (6,6): v_hat[6]*C_alpha2[3][0],
			   (1,1): v_hat[1]*C_alpha2[4][0],
			   (1,3): v_hat[1]*C_alpha2[5][0],
			   (1,5): v_hat[1]*C_alpha2[6][0],
			   (1,7): v_hat[1]*C_alpha2[7][0],
			   (3,1): v_hat[3]*C_alpha2[4][0],
			   (3,3): v_hat[3]*C_alpha2[5][0],
			   (3,5): v_hat[3]*C_alpha2[6][0],
			   (3,7): v_hat[3]*C_alpha2[7][0],
			   (5,1): v_hat[5]*C_alpha2[4][0],
			   (5,3): v_hat[5]*C_alpha2[5][0],
			   (5,5): v_hat[5]*C_alpha2[6][0],
			   (5,7): v_hat[5]*C_alpha2[7][0],
			   (7,1): v_hat[7]*C_alpha2[4][0],
			   (7,3): v_hat[7]*C_alpha2[5][0],
			   (7,5): v_hat[7]*C_alpha2[6][0],
			   (7,7): v_hat[7]*C_alpha2[7][0]}

		"""
		22 multiplications have been cached, 
		will access them to compute components of v_out list
		"""

		v_out=[0 for _ in range(8)]
		v_out[0]=cache[(0,4)]+cache[(2,2)]+cache[(4,4)]+cache[(6,6)]+cache[(1,1)]+cache[(3,3)]+cache[(5,5)]+cache[(7,7)]
		v_out[1]=cache[(0,4)]+cache[(2,6)]-cache[(4,4)]-cache[(6,2)]+cache[(1,3)]-cache[(3,7)]-cache[(5,1)]-cache[(7,5)]
		v_out[2]=cache[(0,4)]-cache[(2,6)]-cache[(4,4)]+cache[(6,2)]+cache[(1,5)]-cache[(3,1)]+cache[(5,7)]+cache[(7,3)]
		v_out[3]=cache[(0,4)]-cache[(2,2)]+cache[(4,4)]-cache[(6,6)]+cache[(1,7)]-cache[(3,5)]+cache[(5,3)]-cache[(7,1)]
		v_out[7]=cache[(0,4)]+cache[(2,2)]+cache[(4,4)]+cache[(6,6)]-cache[(1,1)]-cache[(3,3)]-cache[(5,5)]-cache[(7,7)]
		v_out[6]=cache[(0,4)]+cache[(2,6)]-cache[(4,4)]-cache[(6,2)]-cache[(1,3)]+cache[(3,7)]+cache[(5,1)]+cache[(7,5)]
		v_out[5]=cache[(0,4)]-cache[(2,6)]-cache[(4,4)]+cache[(6,2)]-cache[(1,5)]+cache[(3,1)]-cache[(5,7)]-cache[(7,3)]
		v_out[4]=cache[(0,4)]-cache[(2,2)]+cache[(4,4)]-cache[(6,6)]-cache[(1,7)]+cache[(3,5)]-cache[(5,3)]+cache[(7,1)]

		return v_out
	for i in range(8):
		v_before2[i]=I_1DCT_Chen(A[i])
	v_mid=Transpose(v_before2)
	for i in range(8):
		v_final2[i]=I_1DCT_Chen(v_mid[i])
	return v_final2
'''
C_alpha=[
    [1/2*ncoeff8(i, j)[0]*math.cos(ncoeff8(i,j)[1]*math.pi/16) for j in range(8)]
    for i in range(8)
]
C_coeff=[[ncoeff8(i,j)[0]*ncoeff8(i,j)[1] for j in range(8)] for i in range(8)]
v_before2=[0 for _ in range(8)]
v_final2 = [[0 for _ in range(8)] for _ in range(8)]

print(C_coeff)
'''
print(cache2)
count+=1
'''
def IDCT_Chen2(A):
	"""

	  :param A:
	  :return:
	"""
	cache2 ={}
	def IDCT_Chen_1D(vhat):
		v_out=[0 for _ in range(8)]
		for c in range(8):
			v_temp = 0
			for j in range(8):
				if (j, abs(C_coeff[j][c])) in cache2:
					if C_coeff[j][c]>0:
						v_temp += cache2[(j, abs(C_coeff[j][c]))]
					else:
						v_temp -= cache2[(j, abs(C_coeff[j][c]))]
				else:
					cache2[(j, abs(C_coeff[j][c]))] = vhat[j] * C_alpha[j][c]
					v_temp += cache2[(j, abs(C_coeff[j][c]))]
			v_out[c]=v_temp
		cache2.clear()
		return v_out
	for r in range(8):
		v_before2[r]=IDCT_Chen_1D(A[r])
	v_mid2 = Transpose(v_before2)
	for r in range(8):
		v_final2[r]=IDCT_Chen_1D(v_mid2[r])
	return v_final2
#cache[(r,c,C_alpha2[r][c])]=A[r][c]*C_alpha2[r][c]

idctinput=[[1210.0000000000002, -17.996927353373888, 14.77925623306755, -8.97955730010327, 23.250000000000004, -9.232556779784863, -13.969111825643534, -18.937081397078945],
		   [20.538227580023698, -34.09294403027681, 26.330398745736915, -9.038521236678974, -10.932998995043473, 10.730687995887013, 13.772435090993197, 6.954772758628977],
		   [-10.384029335095121, -23.51407429606327, -1.8535533905932715, 6.040464598200588, -18.07450986222913, 3.196971883068438, -20.417261889578036, -0.8264830874559868],
		   [-8.104806529196079, -5.041456651357671, 14.332151475584148, -14.613395187278673, -8.217831534021599, -2.732392543690965, -3.0845099700035745, 8.429155764630408],
		   [-3.2499999999999862, 9.501426153921093, 7.884638530429614, 1.317025177983375, -11.000000000000002, 17.90447882958578, 18.381919792134465, 15.241160833842573],
		   [3.855633020558628, -2.2146896705631507, -18.16697022287876, 8.499840503342668, 8.268830646150828, -3.608430219669304, 0.8689979628581361, -6.862524463223146],
		   [8.901372633919138, 0.6330199687857636, -2.9172618895780342, 3.6413659899039232, -1.172430484157588, -7.421804998309837, -1.146446609406727, -1.9245633104353126],
		   [0.0491223598079511, -7.81299419433739, -2.424508749415626, 1.5903798159784621, 1.199257102582877, 4.247012669253758, -6.417410588884475, 0.31476943722478534]]

print('IDCT CHEN IMPLMENTATION')
#print(IDCT_Chen(idctinput))
print(IDCT_Chen2(idctinput))
def quantization(A, Q):
	"""

	:param A:
	:param Q:
	:return:
	"""
	quant=[[0 for _ in range(8)] for _ in range(8)]

	for i in range(8):
		for j in range(8):
			quant[i][j] = round(A[i][j]/Q[i][j])
	return quant

def quantizationI(A,Q):
	"""

	:param A:
	:param Q:
	:return:
	"""
	quant = [[0 for _ in range(8)] for _ in range(8)]

	for i in range(8):
		for j in range(8):
			quant[i][j] = round(A[i][j] * Q[i][j])
	return quant


#Luminance and chrominance matrices for Qmatrix function
LQM = [
  [16, 11, 10, 16,  24,  40,  51,  61],
  [12, 12, 14, 19,  26,  58,  60,  55],
  [14, 13, 16, 24,  40,  57,  69,  56],
  [14, 17, 22, 29,  51,  87,  80,  62],
  [18, 22, 37, 56,  68, 109, 103,  77],
  [24, 35, 55, 64,  81, 104, 113,  92],
  [49, 64, 78, 87, 103, 121, 120, 101],
  [72, 92, 95, 98, 112, 100, 103,  99],
]

CQM = [
  [17, 18, 24, 47, 99, 99, 99, 99],
  [18, 21, 26, 66, 99, 99, 99, 99],
  [24, 26, 56, 99, 99, 99, 99, 99],
  [47, 66, 99, 99, 99, 99, 99, 99],
  [99, 99, 99, 99, 99, 99, 99, 99],
  [99, 99, 99, 99, 99, 99, 99, 99],
  [99, 99, 99, 99, 99, 99, 99, 99],
  [99, 99, 99, 99, 99, 99, 99, 99],
]

#################################################################################



def Qmatrix(isY, phi):
	"""

	:param isY:
	:param phi:
	:return:
	"""
	if isY:
		quant=LQM
	else:
		quant=CQM
	if phi>=50:
		for i in range(8):
			for j in range(8):
				quant[i][j]=math.ceil((50+(200-2*phi)*quant[i][j])/100)
	else:
		for i in range(8):
			for j in range(8):
				quant[i][j]=math.ceil((50+round(500/phi)*quant[i][j])/100)
	return quant

#################################################################################

def zigzag(A):
	"""

	:param A:
	:return:
	"""
	m=len(A)
	n=len(A[0])

	for i in range(n+m-1):
		if i%2 == 1:
		#going down and left
			if i<n:
				x=0
				y=i
			else:
				x=i-n+1
				y=n-1
			while x<m and y>=0:
				yield A[x][y]
				x+=1
				y-=1
		else:
		# going up and right
			if i < m:
				x = i
				y = 0
			else:
				x = m - 1
				y = i - m + 1
			while x >= 0 and y < n:
				yield A[x][y]
				x -= 1
				y += 1


matzigzag=[
  [0, 0,  0,  1],
  [0, 0,  7,  8],
  [0,10, 0, 12]
]
g1=zigzag(matzigzag)


def rle0(g):
	"""

	:param g: generator that yields integers
	:return:
	"""
	zero_count=0
	for i in g:
		if i==0:
			zero_count+=1
		else:
			yield zero_count, i
			zero_count=0




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



v=[8, 16, 24, 32, 40, 48, 56, 64]
v2=[101.82, -51.54, -0.0, -5.39, 0.0, -1.61, -0.0, -0.41]





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








mat_sampled=[[2, 3, 4, 2],
			 [2, 3, 4, 2],
			 [1, 2, 2, 1]]




