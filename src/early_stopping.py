class EarlyStopping:
    """
    An example of an early stopping function.

    Args:
        patience (int): How many epochs to wait for a metric to improve.
        minimise (bool): Whether the metric we are monitoring is to minimise,
            such as loss, or maximise, such as accuracy.
    """
    def __init__(
        self,
        patience: int  = 3,
        minimise: bool = True,
        ):
        self.limit    = patience
        self.minimise = minimise
        self.init()

    def improved(self) -> bool:
        """
        Checks whether the last metric has improved.
        """
        if self.minimise:
            return (self._last_loss < self._best_loss)
        else:
            return (self._last_loss > self._best_loss)

    def init(self):
        self._count = 0
        self._best_loss = float("inf" if self.minimise else "-inf")

    def check(self, new: float) -> int:
        """
        Checks a new value of a metric and return a status code
        to indicate whether it has improved since the last check.

        Args:
            new (float): The new value for the metric to check.

        Returns: Status code, either
            0-Improved, 1-Not improve (patience), 2-Not improve (stop).
        """
        self._last_loss = new

        if self.improved():
            self._best_loss = new
            self._count     = 0
            return 0

        self._count += 1
        if self._count <= self.limit:
            print(
                f"Patience {self._count}/{self.limit}, best={self._best_loss}"
            )
            return 1
        else:
            return 2