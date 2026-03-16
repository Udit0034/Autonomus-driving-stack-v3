"""
Lead-vehicle detector using LiDAR point cloud.

Algorithm:
  1. Extract front-region points (x: 0–20 m, y: -2–+2 m)
  2. Simple Euclidean clustering
  3. Report detection flag and distance to nearest cluster centroid
"""

import numpy as np


class VehicleDetector:
    """Detect a lead vehicle from a LiDAR point cloud (numpy array)."""

    def __init__(
        self,
        x_min: float = 1.0,
        x_max: float = 20.0,
        y_min: float = -2.0,
        y_max: float = 2.0,
        cluster_tolerance: float = 1.0,
        min_cluster_size: int = 5,
    ):
        """
        Args:
            x_min: Minimum forward range (m) – excludes ego vehicle hood.
            x_max: Maximum forward range (m).
            y_min: Left lateral bound (m, negative = left).
            y_max: Right lateral bound (m).
            cluster_tolerance: Maximum distance between cluster neighbours (m).
            min_cluster_size: Minimum number of points to count as a cluster.
        """
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.cluster_tolerance = cluster_tolerance
        self.min_cluster_size = min_cluster_size

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, points: np.ndarray) -> tuple:
        """
        Run detection on a point cloud.

        Args:
            points: (N, 3+) array where columns are at least [x, y, z].

        Returns:
            (detected: bool, distance: float)
            distance is NaN when detected is False.
        """
        if points is None or len(points) == 0:
            return False, float('nan')

        # 1. Region-of-interest filter
        roi = self._filter_roi(points)
        if len(roi) < self.min_cluster_size:
            return False, float('nan')

        # 2. Euclidean clustering (greedy, simple)
        clusters = self._cluster(roi[:, :3])
        if not clusters:
            return False, float('nan')

        # 3. Pick the closest cluster centroid
        min_dist = float('inf')
        for cluster in clusters:
            centroid = np.mean(cluster, axis=0)
            dist = np.linalg.norm(centroid[:2])  # distance in xy plane
            if dist < min_dist:
                min_dist = dist

        return True, min_dist

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _filter_roi(self, points: np.ndarray) -> np.ndarray:
        """Return points inside the forward region of interest."""
        mask = (
            (points[:, 0] >= self.x_min)
            & (points[:, 0] <= self.x_max)
            & (points[:, 1] >= self.y_min)
            & (points[:, 1] <= self.y_max)
        )
        return points[mask]

    def _cluster(self, points: np.ndarray) -> list:
        """
        Simple greedy Euclidean clustering.

        Returns a list of numpy arrays, each containing the points of one cluster.
        """
        n = len(points)
        visited = np.zeros(n, dtype=bool)
        clusters: list[np.ndarray] = []

        for i in range(n):
            if visited[i]:
                continue

            # BFS / seed-fill
            queue = [i]
            visited[i] = True
            cluster_indices = []

            while queue:
                idx = queue.pop(0)
                cluster_indices.append(idx)

                diffs = points - points[idx]
                dists = np.linalg.norm(diffs, axis=1)
                neighbours = np.where((dists < self.cluster_tolerance) & ~visited)[0]

                for nb in neighbours:
                    visited[nb] = True
                    queue.append(nb)

            if len(cluster_indices) >= self.min_cluster_size:
                clusters.append(points[cluster_indices])

        return clusters
