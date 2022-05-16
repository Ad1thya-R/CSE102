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

print(RGB2YCbCr(255,255,255))
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
	Y = [[0] * w] * h
	Cb = [[0] * w] * h
	Cr = [[0] * w] * h

	for i, row in enumerate(img):
		for j, pixel in enumerate(row):
			print(pixel.split())
			(r, g, b) = pixel.split()
			(Y[i][j], Cb[i][j], Cr[i][j]) = RGB2YCbCr(r, g, b)
	return (Y, Cb, Cr)

def img_YCbCr2RGB(Y, Cb, Cr):
	'''

	:param Y: 2D array of values of Y for each pixel in image
	:param Cb: 2D array of values of Cb for each pixel in image
	:param Cr: 2D array of values of Cr for each pixel in image
	:return:
	'''
	img=[0*len(Y)]*len(Y[0])
	for i in range(len(Y)):
		for j in range(len(Y[0])):
			converted=YCbCr2RGB(Y[i][j], Cb[i][j], Cr[i][j])
			print(' '.join(str(converted)))
	return img


print(img_RGB2YCbCr([['0 255 0', '0 0 255', '255 255 0'], ['255 255 255', '0 0 0', '0 0 0']]))
print(img_YCbCr2RGB([[255.0, 0.0, 0.0], [255.0, 0.0, 0.0]], [[128.0, 128.0, 128.0], [128.0, 128.0, 128.0]], [[127.99999999999999, 128.0, 128.0], [127.99999999999999, 128.0, 128.0]]))
