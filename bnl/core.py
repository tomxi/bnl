from .formatting import mireval2multi, multi2mireval
import numpy as np
from scipy import stats

class S:
    """A flat segmentation, labeled intervals."""
    def __init__(self, itvls, labels=None, sr=10, Bhat_bw=1, time_decimal=3):
        """Initialize the flat segmentation."""
        self.itvls = itvls
        if labels is None:
            labels = itvls
        self.labels = labels
        self.anno = mireval2multi([itvls], [labels])

        # Build Lstar and T
        self.Lstar = {round(b, time_decimal): l for (b, e), l in zip(itvls, labels)}
        self.beta = np.array(sorted(set(self.Lstar.keys()).union([itvls[-1][-1]])))
        self.T0, self.T = self.beta[0], self.beta[-1]

        # Build BSC and ticks
        self.update_sr(sr)
        self.update_bw(Bhat_bw)

        self.seg_dur = self.beta[1:] - self.beta[:-1]
        self.seg_dur_area_mat = np.outer(self.seg_dur, self.seg_dur)
        self.total_label_agreement_area = np.sum(self.seg_dur_area_mat * self.A(bs=self.beta))

    def update_bw(self, bw):
        """Update bandwidth for Bhat calculation."""
        if hasattr(self, 'Bhat_bw') and self.Bhat_bw == bw:
            return
        self.Bhat_bw = bw
        boundaries = self.beta[1:-1]
        if len(boundaries) == 0:
            self._Bhat = lambda ts: np.array([0 for _ in ts])
        elif len(boundaries) == 1:
            self._Bhat = stats.norm(loc=boundaries[0], scale=bw).pdf
        else:
            kde_bw = bw / boundaries.std(ddof=1) if boundaries.std(ddof=1) != 0 else bw
            kde = stats.gaussian_kde(boundaries, bw_method=kde_bw)
            self._Bhat = kde

    def update_sr(self, sr):
        """Update sampling rate and ticks."""
        if hasattr(self, 'sr') and self.sr == sr:
            return
        self.sr = sr
        self.ticks = np.linspace(self.T0, self.T, int(np.round((self.T - self.T0) * self.sr)) + 1)

    def L(self, x):
        """Return the label for a given time x."""
        if not (self.T0 <= x <= self.T):
            raise IndexError(f'RANGE: {x} outside the range of this segmentation!')
        idx = np.searchsorted(self.beta, x, side='right') - 1
        return self.Lstar[self.beta[idx]]

    def B(self, x):
        """Return whether x is a boundary."""
        return int(x in self.beta)

    def Bhat(self, ts=None):
        """Return the boundary salience curve at given time steps."""
        if ts is None:
            ts = self.ticks
        return self._Bhat(ts)

    def A(self, bs=None):
        """Return the label agreement indicator for given boundaries."""
        if bs is None:
            bs = self.beta
        ts = (bs[1:] + bs[:-1]) / 2  # Sample label from mid-points of each frame
        sampled_anno = self.anno.to_samples(ts)
        sample_labels = [obs[0]['label'] for obs in sampled_anno]
        return np.equal.outer(sample_labels, sample_labels).astype(float)

    def plot(self, **kwargs):
        pass  # Placeholder for plotting functionality

class H:
    """A hierarchical segmentation composed of multiple flat segmentations."""
    def __init__(self, itvls, labels=None, sr=10, Bhat_bw=1, time_decimal=3):
        pass  # Placeholder for hierarchical segmentation initialization

    def update_sr(self, sr):
        pass  # Placeholder for updating sampling rate

    def update_bw(self, Bhat_bw):
        pass  # Placeholder for updating bandwidth

    def A(self, bs=None):
        pass  # Placeholder for label agreement calculation

    def B(self):
        pass  # Placeholder for boundary checking

    def plot(self, **kwargs):
        pass  # Placeholder for plotting functionality