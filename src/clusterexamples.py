from src.clusterer import Clusterer

year = input('Enter year: ')
pos = input('Enter position: ')

clusterer = Clusterer(year, pos)
clusters = clusterer.make_clusters()
for i in range(1, len(clusters) + 1):
    print("Cluster {}: ".format(i))
    print(clusters[i])
