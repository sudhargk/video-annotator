from PIL import Image
import random


class JSEG():
	def __init__(self, path):
		self.path = path
		self.img = Image.open(path)
		self.cluster = Image.open(path)
		self.w, self.h = self.img.size

	def cluster_colors(self, segments=3, passes=2):
		"""
		k-means clustering
		:param segments: number of segments to create
		:param passes: number of passes to make in getting means
		"""
		param = 0.01
		means = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _ in range(segments)]
		img = self.cluster.load()
		for _ in range(passes):
			for x in range(self.w):
				for y in range(self.h):
					closest_k = None
					smallest_error = None
					for i, m in enumerate(means):
						error = abs(img[x, y][0] - m[0]) + abs(img[x, y][1] - m[1]) + abs(img[x, y][2] - m[2])
						if smallest_error is None or error < smallest_error:
							smallest_error = error
							closest_k = i
					# compute new means for closest k
					means[closest_k] = (means[closest_k][0] * (1 - param) + img[x, y][0] * param,
										means[closest_k][1] * (1 - param) + img[x, y][1] * param,
										means[closest_k][2] * (1 - param) + img[x, y][2] * param,)

		out = Image.new('I', self.img.size, 0xffffff)
		for x in range(self.w):
			for y in range(self.h):
				closest_k = None
				smallest_error = None
				for i, m in enumerate(means):
					error = abs(img[x, y][0] - m[0]) + abs(img[x, y][1] - m[1]) + abs(img[x, y][2] - m[2])
					if smallest_error is None or error < smallest_error:
						smallest_error = error
						closest_k = i
				# change pixel

				img[x, y] = (int(means[closest_k][0]), int(means[closest_k][1]), int(means[closest_k][2]))

		index_period = self.path.rfind('.')
		self.cluster.save(self.path[0:index_period] + '_clustered' + self.path[index_period:])


def main():
	import sys
	img = JSEG(sys.argv[1])
	img.cluster_colors(10)


if __name__ == "__main__":
	main()
