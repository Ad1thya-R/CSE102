'''
The goal of this project is to develop a lossy image compression
format close to JPEG. Lossy compression algorithms allow to
greatly reduce the size of image files at the price of loosing some data from the original image. In general, a lossy image compression works by pruning the information the human eye is not sensible to. This is the case, for instance, in the JPEG file format.
'''


def ppm_tokenize(stream):
	'''

	:param stream: input stream such as g=open('file')
	:return: iterator (generator) that outputs the data individually
	'''
	for line in stream:
		line = line.partition('#')[0]
		line = line.rstrip().split()
		if line:
			yield line



inp=open('file')

g=ppm_tokenize(inp)
tokens=[]

def ppm_load(stream):
	'''

	:param stream: input stream such as g=open('file')
	:return: width, height and 2D-array for image
	'''
	g=ppm_tokenize(stream)
	img=[]
	for i in range(3):
		curr=next(g)
		if i==1:
			w = curr[0]
			h = curr[1]
	for i in g:
		curr=i
		img.append(curr)
	return (w,h,img)


def ppm_save(w,h,img, output):
	'''takes an output stream output and
	that saves the PPM image img whose size is w x h.'''
	pass





print(ppm_load(inp))