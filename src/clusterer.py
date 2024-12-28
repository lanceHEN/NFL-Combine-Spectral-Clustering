import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.preprocessing import scale

class Clusterer:
    """
    The Clusterer class is responsible for clustering NFL players for a given position, for a given draft year, according
    to their metrics in the combine.

    Attributes:
        df_pos (pd.DataFrame): A DataFrame containing player data for a specific position.
    """

    def __init__(self, year, pos):
        """
        Initializes the Clusterer with player data for the specified year and position.

        Args:
            year (int): The year of the draft data to load. Must be 2020 or after.
            pos (str): The position of players to filter (e.g., 'QB', 'RB').

        Raises:
            ValueError: If the year is invalid (pre 2020) or the position is not recognized.
        """
        try:
            df_all = pd.read_csv('data/draft' + str(year) + '.csv')
        except:
            raise ValueError('Year not recognized: {}'.format(year))

        # remove all entries that don't have a height
        df_all = df_all.dropna(subset = ['Ht'])

        # convert height to total inches
        def height_to_inches(height):
            feet, inches = map(int, height.split('-'))
            total_inches = feet * 12 + inches
            return total_inches

        df_all['Ht'] = df_all['Ht'].apply(height_to_inches)

        if pos not in set(df_all['Pos']):
            raise ValueError('Position not recognized: {}'.format(pos))
        # only get the numerical figures
        self.df_pos = df_all[df_all['Pos'] == pos][['Player','Ht','Wt','40yd','Vertical','Broad Jump', '3Cone',
                                                    'Shuttle']]

    @staticmethod
    def __impute(df):
        """
        Imputes missing values in the DataFrame using K-Nearest Neighbors.

        Args:
            df (pd.DataFrame): DataFrame with missing values.

        Returns:
            pd.DataFrame: DataFrame with imputed values.
        """
        imputer = KNNImputer()
        df_imputed = pd.DataFrame(imputer.fit_transform(df))
        return df_imputed

    @staticmethod
    def __get_scaled_arr(df):
        """
        Converts a pandas dataframe to a scaled np array.

        Args:
            df (pd.DataFrame): dataframe to scale.

        Returns:
            np.ndarray: Scaled np array.
        """
        arr = df.to_numpy()
        return scale(arr)

    def make_clusters(self, k=None):
        """
        Creates clusters of players using spectral clustering with Gaussian RBF kernel.

        Args:
            k (int, optional): The number of clusters to form. If None, the optimal number is determined using the
            elbow method.

        Returns:
            pd.Series: A Series of clusters with player names grouped by cluster.

        Raises:
            TypeError: If k is not an integer.
            ValueError: If k is less than 2.
        """
        # get rid of names for calculations
        df_pos_numerical = self.df_pos.loc[:, self.df_pos.columns != 'Player']
        # impute by replacing with means
        pos_imputed = Clusterer.__impute(df_pos_numerical)
        # convert to scaled np array, X
        X = Clusterer.__get_scaled_arr(pos_imputed)
        # obtain gamma by computing median pairwise dist
        γ = Clusterer.__calculate_gamma(X)
        # obtain gaussian rbf kernel matrix
        K = Clusterer.__gaussian_rbf_mat(X, γ=γ)

        if k is None:
            k = Clusterer.__optimal_k_elbow(X, K)
        else: #ensure supplied k is valid: int and >= 2
            if not isinstance(k, int):
                raise TypeError('k must be an integer')
            if k <= 1:
                raise ValueError('k must be greater than 1')

        # project data onto U
        U = np.linalg.eig(K)[1][:,:k]
        Ū = np.apply_along_axis(lambda x: x / np.linalg.norm(x, ord=2), axis=0, arr=U)
        if k is None:
            k = Clusterer.__optimal_k_elbow(Ū)
        self.df_pos['Cluster'], clusters, centroids = Clusterer.__kmeans(Ū,k)

        # Group players by their cluster and extract names
        clustered_players = self.df_pos.groupby('Cluster')['Player'].apply(list)

        return clustered_players

    @staticmethod
    def __calculate_gamma(X):
        """
        Calculates the gamma parameter for the RBF kernel using the median pairwise distance.

        Args:
            X (np.ndarray): Data array.

        Returns:
            float: Gamma parameter.
        """
        pairwise_dists = []

        n = X.shape[0]
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.linalg.norm(X[i] - X[j],ord=2)
                pairwise_dists.append(dist)

        σ = np.median(pairwise_dists)
        return 1 / (2 * σ **2)

    @staticmethod
    def __gaussian_rbf(xi,xj,γ):
        """
        Computes the Gaussian RBF kernel between two points.

        Args:
            xi (np.ndarray): First data point.
            xj (np.ndarray): Second data point.
            γ (float): Gamma parameter.

        Returns:
            float: RBF kernel value.
        """
        return np.exp(-γ * (np.linalg.norm(xi - xj, ord=2))**2)

    @staticmethod
    def __gaussian_rbf_mat(X, γ):
        """
        Computes the Gaussian RBF kernel matrix for a dataset.

        Args:
            X (np.ndarray): Data array.
            γ (float): Gamma parameter.

        Returns:
            np.ndarray: RBF kernel matrix.
        """
        n = X.shape[0]
        d = X.shape[1]
        K = np.zeros((n,n))
        for i, xi in enumerate(X):
            for j, xj in enumerate(X):
                xi = np.reshape(xi, (d,1))
                xj = np.reshape(xj, (d,1))
                K[i,j] = Clusterer.__gaussian_rbf(xi, xj, γ)
        return K

    @staticmethod
    def __kmeans(X, k):
        """
        Performs k-means clustering on the dataset.

        Args:
            X (np.ndarray): Data array.
            k (int): Number of clusters.

        Returns:
            tuple: A tuple containing predictions, clusters, and centroids.
        """
        # initialize centroids to k random values in X
        np.random.seed(42)
        rand_indices = np.random.choice(range(X.shape[0]), k, replace=False)
        centroids = X[rand_indices]
        clusters = [[] for _ in range(k)]
        converged = False
        while not converged:
            # reset clusters
            clusters = [[] for _ in range(k)]

            # assign each point to cluster whose centroid is closest
            for x in X:
                distances = [np.linalg.norm(x - centroid, ord=2) for centroid in centroids]
                closest_centroid_idx = np.argmin(distances)
                clusters[closest_centroid_idx].append(x)

            old_centroids = centroids.copy()

            # calculate new centroids using points in each cluster
            centroids = np.array([np.mean(cluster, axis=0) if cluster else old_centroids[idx]
                              for idx, cluster in enumerate(clusters)])

            converged = np.array_equal(old_centroids, centroids)

        # return solutions in order
        predictions = np.zeros(X.shape[0], dtype=int)

        for idx, x in enumerate(X):
            distances = [np.linalg.norm(x - centroid, ord=2) for centroid in centroids]
            predictions[idx] = np.argmin(distances)

        predictions = np.array(predictions)

        return predictions, clusters, centroids

    @staticmethod
    def __optimal_k_elbow(X, K, max_clusters=10):
        """
        Determines the optimal number of clusters using the elbow method.

        Args:
            X (np.ndarray): Data array.
            K (np.ndarray): Kernel matrix.
            max_clusters (int): Maximum number of clusters to consider.

        Returns:
            int: Optimal number of clusters.
        """
        dist_sums = []

        for k in range(2, max_clusters + 1):
            U = np.linalg.eig(K)[1][:,:k]
            Ū = np.apply_along_axis(lambda x: x / np.linalg.norm(x, ord=2), axis=0, arr=U)
            predictions, clusters, centroids = Clusterer.__kmeans(Ū,k)
            dist_sum = 0
            for cluster, centroid in zip(clusters, centroids):
                for pt in cluster:
                    dist_sum += (np.linalg.norm(pt - centroid, ord=2))**2

            dist_sums.append(dist_sum)

        # get second derivative of sums (minimum will be elbow)
        second_deriv = np.diff(np.diff(dist_sums))

        # add 2 because diff reduces indices
        return np.argmin(second_deriv) + 2