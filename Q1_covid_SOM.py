import csv
import random
import math
from PIL import Image
import geopandas as gpd
import matplotlib.pyplot as plt


class SOM:

    def __init__(self, n=10, alpha=0.5, epoch=100, decay_L=1, decay_N=1):
        self.data = dict()
        self.weights = []
        self.n = n
        self.alpha = alpha
        self.epoch = epoch

        self.confirmed = 0
        self.death = 0
        self.recovered = 0

        self.decay_L = decay_L
        self.decay_N = decay_N

    def readData(self, path):
        with open(path, newline='') as csvfile:
            reader = csv.reader(csvfile)
            c = 0
            for row in reader:
                if c != 0:
                    if self.confirmed < int(row[5]):
                        self.confirmed = int(row[5])
                    if self.death < int(row[6]):
                        self.death = int(row[6])
                    if self.recovered < int(row[7]):
                        self.recovered = int(row[7])

                    if row[1] not in list(self.data.keys()):
                        self.data[row[1]] = [
                            int(row[5]), int(row[6]),
                            int(row[7])
                        ]
                    else:
                        temp = [
                            self.data[row[1]][0] + int(row[5]),
                            self.data[row[1]][1] + int(row[6]),
                            self.data[row[1]][2] + int(row[7])
                        ]
                        self.data[row[1]] = temp
                c += 1

    def distance(self, neuron, inp):
        current = self.Weights[neuron]
        total = 0
        for el in range(len(inp)):
            total += (inp[el] - current[el])**2
        return total**(1 / 2)

    def initialize(self):
        for i in range(self.n):
            row = []
            for j in range(self.n):
                c = random.randint(0, self.confirmed + 1)
                d = random.randint(0, self.death + 1)
                r = random.randint(0, self.recovered + 1)
                row.append([c, d, r])
            self.weights.append(row)

    def neighborhood(self, target, current, epoch):
        if self.decay_L == 1:
            return (1 / (math.sqrt((target[1] - current[1])**2 +
                                   (target[0] - current[0])**2))) * (math.exp(
                                       -epoch / self.epoch))
        else:
            return (1 / (math.sqrt((target[1] - current[1])**2 +
                                   (target[0] - current[0])**2))) * (
                                       (self.epoch - epoch) / self.epoch)

    def L(self, epoch):
        if self.decay_L == 1:
            return self.alpha * (math.exp(-epoch / self.epoch))
        else:
            return self.alpha * ((self.epoch - epoch) / self.epoch)

    def update(self, neuron, inp, epoch):
        for i in range(self.n):
            for j in range(self.n):

                if neuron[0] == i and neuron[1] == j:
                    self.weights[i][j][0] = self.weights[i][j][0] + (
                        self.L(epoch) * (inp[0] - self.weights[i][j][0]))
                    self.weights[i][j][1] = self.weights[i][j][1] + (
                        self.L(epoch) * (inp[1] - self.weights[i][j][1]))
                    self.weights[i][j][2] = self.weights[i][j][2] + (
                        self.L(epoch) * (inp[2] - self.weights[i][j][2]))
                else:
                    self.weights[i][j][0] = self.weights[i][j][0] + (
                        self.L(epoch) *
                        self.neighborhood(neuron, [i, j], epoch) *
                        (inp[0] - self.weights[i][j][0]))
                    self.weights[i][j][1] = self.weights[i][j][1] + (
                        self.L(epoch) *
                        self.neighborhood(neuron, [i, j], epoch) *
                        (inp[1] - self.weights[i][j][1]))
                    self.weights[i][j][2] = self.weights[i][j][2] + (
                        self.L(epoch) *
                        self.neighborhood(neuron, [i, j], epoch) *
                        (inp[2] - self.weights[i][j][2]))

    def get_BMU(self, inp):
        best = math.inf
        BMU = [0, 0]
        for i in range(self.n):
            for j in range(self.n):
                d = math.sqrt((self.weights[i][j][0] - inp[0])**2 +
                              (self.weights[i][j][1] - inp[1])**2 +
                              (self.weights[i][j][2] - inp[2])**2)
                if d <= best:
                    best = d
                    BMU = [i, j]
        return BMU

    def learn(self, path, show=False, stamp=10):
        self.readData(path)
        countries = list(self.data.keys())
        self.initialize()
        for e in range(self.epoch):
            dataPoint = self.data[random.choice(countries)]
            BMU = self.get_BMU(dataPoint)
            self.update(BMU, dataPoint, e)
            if show == True and e % stamp == 0:
                self.showColors()
        self.showColors()
        self.worldMap()

    def display(self):
        for i in self.weights:
            print(i)

    def colors(self):
        colorsMap = []
        for i in range(self.n):
            row = []
            for j in range(self.n):
                r = 255 * (self.weights[i][j][0] / self.confirmed)
                g = 255 * (self.weights[i][j][1] / self.death)
                b = 255 * (self.weights[i][j][2] / self.recovered)
                row.append((int(r), int(g), int(b)))
            colorsMap.append(row)
        return colorsMap

    def showColors(self):
        colorMap = self.colors()
        height = 100
        width = 100
        pixels = []
        for i in range(height):
            row = []
            for j in range(width):
                row.append((0, 0, 0))
            pixels.append(row)

        for i in range(self.n):
            for j in range(self.n):
                for l in range(10):
                    for k in range(10):
                        pixels[(i * 10) + l][(j * 10) + k] = colorMap[i][j]
        pixelsFlattened = []
        for r in pixels:
            for e in r:
                pixelsFlattened.append(e)

        img = Image.new("RGB", (height, width))

        img.putdata([rgb for rgb in pixelsFlattened])
        img.show()

    def worldMap(self):

        world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
        colorMap = self.colors()

        worldColors = {}
        for c in list(self.data.keys()):
            BMU = self.get_BMU(self.data[c])
            color = colorMap[BMU[0]][BMU[1]]
            worldColors[c] = color

        def get_color_rgb(row):
            country_name = row['name']
            if country_name in worldColors:
                return worldColors[country_name]
            else:
                return (128, 128, 128)

        world['color_rgb'] = world.apply(get_color_rgb, axis=1)

        world['color'] = world['color_rgb'].apply(
            lambda x: '#%02x%02x%02x' % x)

        fig, ax = plt.subplots(figsize=(10, 6))
        world.plot(ax=ax, color=world['color'], edgecolor='black')

        plt.title('World Map with Custom RGB Colors')

        plt.show()


som = SOM(n=10, alpha=0.2, epoch=10000, decay_L=1, decay_N=1)

som.learn('Q1_countrydata.csv', True, 1000)
