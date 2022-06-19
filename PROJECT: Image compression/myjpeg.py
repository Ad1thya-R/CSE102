"""
The goal of this project is to develop a lossy image compression
format close to JPEG. Lossy compression algorithms allow to
greatly reduce the size of image files at the price of loosing some data from
the original image. In general, a lossy image compression works by pruning the
information the human eye is not sensible to. This is the case, for instance,
in the JPEG file format.
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
	#We go through each line in the file
	for line in stream:
		#Remove the comments from the file in this manner (assume comments are made after typing the content on each line
		line = line.partition('#')[0]

		#Remove any space at the end of the string and split the elements of the line to obtain a list
		line = line.rstrip().split()

		#Check that the line is non-empty and iterate
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
	#iterate through the first 4 elements of the stream to find width and height
	for i in range(4):
		curr=next(g)
		if i==1:
			w = curr
		if i==2:
			h = curr

	#go through remaining elements in ppm_tokenize and add different colors as lists of [r,g,b] form
	for i in range(int(h)):
		temprow=[]
		for j in range(int(w)):
			curr2=[int(next(g)) for _ in range(3)]
			temprow.append(tuple(curr2))
		img.append(temprow)
	return int(w),int(h),img

def ppm_save(w: int, h: int, img: list, output) -> None:
	"""takes an output stream output and
	that saves the PPM image img whose size is w x h."""
	#Write the data to produce a file of PPM format
	output.write('P3\n')
	output.write(f'{w} {h}\n')
	output.write('255\n')
	for row in img:
		for rgb in row:
				output.write(f'{rgb[0]} {rgb[1]} {rgb[2]}\n')



def RGB2YCbCr(r: int, g: int, b: int) -> tuple:
	"""
	:int r: red, 0 to 255
	:int g: green 0 to 255
	:int b: blue 0 to 255
	:return: Y, Cb, Cr color format
	"""
	#Use the formulas relating YCbCr and RGB
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
	rgb=[r,g,b]
	##Make sure output lies between 0 and 255
	for i in range(len(rgb)):
		if rgb[i]<0:
			rgb[i]=0
		elif rgb[i]>255:
			rgb[i]=255
	return (round(rgb[0]), round(rgb[1]), round(rgb[2]))



def img_RGB2YCbCr(img: list) -> list:
	"""
	:param img: 2D array with pixels stored as rgb values ('255 255 255')
	:return: Y[i][j], Cb[i][j], Cr[i][j] --> 3 matrices with these attributes
	and i,j denoting the position of the pixel
	"""
	h=len(img)
	w=len(img[0])
	#initialise matrices for Y, Cb, Cr
	Y=[]
	Cb=[]
	Cr=[]
	#Iterate through img and fill Y, Cb, Cr matrices using converter function above
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
	#Use Y, Cb, Cr matrices to convert back to rgb pixels and fill a singl img matrix
	img=[]
	for i in range(len(Y)):
		row=[]
		for j in range(len(Y[0])):
			r,g,b=YCbCr2RGB(Y[i][j], Cb[i][j], Cr[i][j])
			row.append(str(r)+' '+str(g)+' '+str(b))
		img.append(row)
	return img

def subsampling(w : int, h: int, C: list, a: int, b: int) -> list:
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


	img = []
	#split matrix by height by stepping by a through the rows of the matrix
	for i in range(0, h, a):
		row = []
		C_cut=C[i:i+a]
		#split matrix by width by stepping by b through the columns of the cut-by-height matrix
		for j in range(0, w, b):
			temp_sum=0
			for c in C_cut:
				temp_sum+=sum(c[j:j+b])
				temp_len=len(c[j:j+b])
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

	#Check if input C is matrix
	for row in C:
		if len(C[0])!=len(row):
			raise InvalidInput(C, "A matrix of integers (A 2D array where the rows are of equal length).")
	#inputs a and b cannot be bigger than the dimensions of the matrix
	if a>w or b>h:
		raise InvalidInput((a,b),"integers a and b s.t. a<=w and b<=h")

	##Check whether 2D array is a matrix
	for row in C:
		if len(C[0])!=len(row):
			raise InvalidInput(C, "A matrix of integers (A 2D array where the rows are of equal length).")

	img=[]
	for smallrow in C:
		temp_row=[]
		for pixel in smallrow:
			#Multiply pixels by a to extrapolate
			temp_row+=[pixel]*a
			temp_row=temp_row[:w]
		#Multiply the obtained row by b to obtain the a*b block that we desired
		img.extend([temp_row]*b)
	img=img[:h]
	return img

def block_splitting(w: int, h: int, C: list) -> list:
	"""

	:param w: The width of the entire block (no. of columns)
	:param h: The height of the entire block (no. of rows)
	:param C: Matrix of size w x h
	:return: yields blocks of size 8x8 one by one
	"""
	#Check if input C is matrix
	for row in C:
		if len(C[0])!=len(row):
			raise InvalidInput(C, "A matrix of integers (A 2D array where the rows are of equal length).")
	#Check if dimensions match input array
	if h!=len(C) or w!=len(C[0]):
		raise InvalidInput(f'{(w,h)} are not the dimensions of the input matrix, it is', f'please input the correct dimensions of the matrix img, {len(C)}x{len(C[0])}')


	#Split blocks by height
	for i in range(0, h, 8):
		block = []
		#Check whether last rows of blocks are of height less than 8 (to extrapolate it)
		if i + 8 > h:
			temp_col = C[i:i + 8]
			t = len(temp_col)
			temp_col.extend([temp_col[t - 1]] * (8 - t))
			C_cut=temp_col
		else:
			C_cut=C[i:i+8]
		#Split blocks by width
		for j in range(0, w, 8):
			for c in C_cut:
				if j + 8 > w:
					temp_row=c[j:j+8]
					t=len(temp_row)
					temp_row.extend([temp_row[t-1]]*(8-t))
					block.append(temp_row)
				else:
					block.append(c[j:j+8])
		#yield blocks one by one
		for y in range(0,len(block),8):
			yield block[y:y+8]


def Cn(n: int) -> int:
	"""

	:param n:The size of Cn matrix needed for DCT and IDCT naive algorithms
	:return: Cn matrix
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

	:param n: The size of Cn matrix transposed needed for DCT and IDCT naive algorithms
	:return: Transpose of the Cn matrix
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

	:param v: A list of floats
	:return: A list of floats with the naive 1 dimensional DCT implemented on it
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

	:param v: A list of floats
	:return: A list of floats with the naive 1 dimensional IDCT implemented on it
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
	"""

	:param A: matrix of size a x n
	:param B: matrix of size n x b
	:return: the product of the matrices A and B
	"""
	# we assume that A and B are non - empty matrices
	m = len ( A )
	n = len ( A [0])
	if len ( B ) != n :
		return None # the size does not match

	#naive matrix multiplication algorithm
	p = len ( B [0])
	C = [ None ] * m
	for i in range ( m ):
		C [ i ] = [0] * p
		for k in range ( p ):
			for j in range ( n ):
				C[i][k] += A[i][j] * B[j][k]
	return C

def Transpose(A: list) -> list:
	"""

	:param A: A matrix of size a x n
	"""
	return [[A[j][i] for j in range(len(A))] for i in range(len(A[0]))]

def DCT2(m: int, n: int, A: list):
	"""

	:param m: height (no of rows)
	:param n: width (no of columns)
	:param A: input matrix
	:return: A list of floats with the naive 2 dimensional DCT implemented on it
	"""
	# Check if input A is matrix
	for row in A:
		if len(A[0]) != len(row):
			raise InvalidInput(A, "A matrix of integers (A 2D array where the rows are of equal length).")
	# Check if dimensions match input array
	if m != len(A) or n != len(A[0]):
		raise InvalidInput(f'{(m,n)} are not the dimensions of the input matrix, it is', f'please input the correct dimensions of the matrix A, {len(A)}x{len(A[0])}')

	#code for function
	CnT=Cn_T(n)
	Cm=Cn(m)
	first_mult=matProd(A,CnT)
	return matProd(Cm,first_mult)

def IDCT2(m: int, n: int, A: list):
	"""

	:param m: height (no of rows)
	:param n: width (no of columns)
	:param A: input matrix
	:return: A list of floats with the naive 2 dimensional IDCT implemented on it
	"""
	# Check if input A is matrix
	for row in A:
		if len(A[0]) != len(row):
			raise InvalidInput(A, "A matrix of integers (A 2D array where the rows are of equal length).")
	# Check if dimensions match input array
	if m != len(A) or n != len(A[0]):
		raise InvalidInput(f'{(m, n)} are not the dimensions of the input matrix, it is',f'please input the correct dimensions of the matrix A, {len(A)}x{len(A[0])}')

	#code for function
	C = Cn(n)
	Cm = Cn_T(m)
	first_mult = matProd(A, C)
	return matProd(Cm, first_mult)

def redalpha(i: int) -> tuple:
	"""

	:param i: integer
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

def DCT_Chen(A):
	"""

	:param A: Input matrix to be converted using Chen's algorithm (352 multiplications)
	:return: DCT matrix
	"""
	#Check if input A is matrix
	for row in A:
		if len(A[0]) != len(row):
			raise InvalidInput(A, "A matrix of integers (A 2D array where the rows are of equal length).")

	#Apply 1DCT transform on rows
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
	#Transpose intermediary result to allow to apply the transform on columns
	v_mid=Transpose(v_before)
	#Apply 1DCT transform on the columns
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



##pre-compute C_alpha matrix\
C_alpha=[
    [1/2*ncoeff8(i, j)[0]*math.cos(ncoeff8(i,j)[1]*math.pi/16) for j in range(8)]
    for i in range(8)
]
C_coeff=[[ncoeff8(i,j)[0]*ncoeff8(i,j)[1] for j in range(8)] for i in range(8)]
v_before2=[0 for _ in range(8)]
v_final2 = [[0 for _ in range(8)] for _ in range(8)]


def IDCT_Chen2(A):
	"""

	  :param A: Input matrix to invert DCT using Chen's algorithm (352 multiplications)
	  :return:
	"""
	#Check if input A is matrix
	for row in A:
		if len(A[0]) != len(row):
			raise InvalidInput(A, "A matrix of integers (A 2D array where the rows are of equal length).")

	#dictionary to store multiplications so that repeated multiplications can be accessed:
	cache2 ={}

	#1D Transform
	def IDCT_Chen_1D(vhat):
		v_out=[0 for _ in range(8)]
		for c in range(8):
			v_temp = 0
			for j in range(8):
				if (j, abs(C_coeff[j][c])) in cache2:
					#avoid storing negative of same multiplication
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
	#Apply 1DCT on the rows
	for r in range(8):
		v_before2[r]=IDCT_Chen_1D(A[r])

	#Transpose the matrix to apply the 1DCT again to the columns of the matrix
	v_mid2 = Transpose(v_before2)
	for r in range(8):
		v_final2[r]=IDCT_Chen_1D(v_mid2[r])
	return v_final2


def quantization(A, Q):
	"""

	:param A: Input matrix to quantize
	:param Q: Quantization matrix
	:return: Quantized matrix
	"""
	quant=[[0 for _ in range(8)] for _ in range(8)]

	for i in range(8):
		for j in range(8):
			quant[i][j] = round(A[i][j]/Q[i][j])
	return quant

def quantizationI(A,Q):
	"""

	:param A: Input matrix to quantize
	:param Q: Inverse quantization matrix
	:return: Inverse quantized matrix
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

	:param isY: Boolean to determine whether luminance or chrominance matrix is to be used
	:param phi: Quality factor
	:return: Qmatrix
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

	:param A: Input matrix
	:return: Zigzag traversal of matrix as a generator
	"""
	#Check if input A is matrix
	for row in A:
		if len(A[0]) != len(row):
			raise InvalidInput(A, "A matrix of integers (A 2D array where the rows are of equal length).")
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




def rle0(g):
	"""

	:param g: generator that yields integers
	:return: generator that yields the pairs obtained from the RLE0 encoding of g
	"""
	zero_count=0
	for i in g:
		if i==0:
			zero_count+=1
		else:
			yield zero_count, i
			zero_count=0




