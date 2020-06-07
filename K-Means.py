import numpy as np
import sys
import re


class Preprocess:
    def __init__(self, data_path):
        self.data = open(data_path, 'r').read()

    def __remove_hashtag(self, text):
        return text.replace('#', '')

    def __remove_tagged_person(self, text):
        start = 0
        while start != -1:
            start = text.find('@')
            end = start
            while end < len(text) and text[end] is not None and text[end] != ' ':
                end += 1
            tagged_person = text[start: end]
            text = text.replace(tagged_person, '')
        return text

    def __remove_url(self, text):
        return re.sub(r'http\S+', '', text)

    def preprocess_data(self):
        inputs = self.data.splitlines()
        results = []
        for i in range(len(inputs)):
            split = inputs[i].split('|')
            text = self.__remove_hashtag(split[2])
            text = self.__remove_tagged_person(text)
            text = self.__remove_url(text)
            text = text.strip().lower()
            text = " ".join(text.split())
            results.append(text)

        return results


class TweetsClustering:
    def __init__(self, k, data):
        self.k = k
        self.data = data
        self.clusters = {}
        self.centroids = {}

    def predict(self):
        self.__init_centroid()
        convergence = False
        iteration = 0
        print("--------------------------------------------")
        while True:
            print("iteration " + str(iteration))
            old_centroids = self.centroids.copy()
            self.__build_cluster()
            self.__update_centroids()
            count = 0
            for i in range(len(self.centroids)):
                if self.centroids[i + 1] == old_centroids[i + 1]:
                    count += 1
                if count == len(self.centroids):
                    convergence = True
            if iteration == 50:
                print("exceed 50 iteration, we stop here")
                convergence = True
            print("--------------------------------------------")
            if convergence:
                break
            iteration += 1
        for index, cluster in self.clusters.items():
            print("cluster" + str(index) + " :")
            # for i in range(len(cluster)):
            #     print(str(cluster[i]))
            print(str(index) + ": " + str(len(cluster)) + " tweets")
            print("--------------------------------------------")

        self.__calculate_sse()

    def __build_cluster(self):
        clusters = {}
        for index in range(self.k):
            clusters[index + 1] = []

        for i in range(len(self.data)):
            min_distance = sys.maxsize
            cluster_index = 1
            for index, centroid in self.centroids.items():
                distance = self.__jaccard_distance(self.data[i], centroid)
                if distance < min_distance:
                    min_distance = distance
                    cluster_index = index
            clusters[cluster_index].append(self.data[i])

        self.clusters = clusters

    def __init_centroid(self):
        self.centroids = {
            i + 1: np.random.choice(data, self.k)[i]
            for i in range(self.k)
        }

    def __update_centroids(self):
        centroids = {}
        min_distance = sys.maxsize
        for index, tweet_cluster in self.clusters.items():
            new_centroid = tweet_cluster[0]
            print("updating cluster" + str(index) + "'s centroid")
            for tweet_a in tweet_cluster:
                distance = 0.0
                for tweet_b in tweet_cluster:
                    distance += self.__jaccard_distance(tweet_a, tweet_b)
                if distance < min_distance:
                    min_distance = distance
                    new_centroid = tweet_a
            centroids[index] = new_centroid

        self.centroids = centroids

    def __jaccard_distance(self, tweet_a, tweet_b):
        split_a = tweet_a.split(' ')
        split_b = tweet_b.split(' ')
        tweet_union = len(set(split_a).union(set(split_b)))
        tweet_intersection = len(set(split_a).intersection(set(split_b)))
        distance = 1.0 - tweet_intersection/tweet_union
        return distance

    def __calculate_sse(self):
        sse = 0.0
        for i in range(self.k):
            centroid = self.centroids[i + 1]
            for tweet in self.clusters[i + 1]:
                distance = self.__jaccard_distance(tweet, centroid)
                sse += distance ** 2
        print("SSE: " + str(sse))
        return sse


if __name__ == "__main__":

    if len(sys.argv) != 2:
        print("Please input the correct arguments!")
        exit()

    preprocess = Preprocess('everydayhealth.txt')
    data = preprocess.preprocess_data()
    k = int(sys.argv[1])
    print("Total Cluster: " + str(k))
    tweet_clustering = TweetsClustering(k, data)
    tweet_clustering.predict()


