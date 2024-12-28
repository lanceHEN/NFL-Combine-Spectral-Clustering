from src.clusterer import Clusterer

year = input('Enter year: ')
pos = input('Enter position: ')

clusterer = Clusterer(year, pos)
clusters = clusterer.make_clusters()
for i in range(len(clusters)):
    print("Cluster {}: ".format(i + 1))
    print(clusters[i])