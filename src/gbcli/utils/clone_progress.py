"""Clone progress reporter, kept apart from ``utils`` so gitpython loads only on clone."""

from git import RemoteProgress
from tqdm import tqdm


class CloneProgress(RemoteProgress):
    def __init__(self, update_bar):
        super().__init__()
        self.update_bar = update_bar
        if not update_bar:
            self.pbar = tqdm(leave=False)

    def update(self, op_code, cur_count, max_count=None, message=""):
        if self.update_bar:
            # convert to step size for specified total
            step_size = 100 / max_count
            self.update_bar(
                callback_event="preparing_contents", callback_args={"steps": step_size}
            )
        else:
            self.pbar.total = max_count
            self.pbar.n = cur_count
            self.pbar.refresh()
