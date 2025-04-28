def get_decayed_lr(initial_lr, lr_decay, epoch):
    """
    Calculate the decayed learning rate based on the current epoch.

    Parameters
    ----------
    initial_lr : float
        The initial learning rate before decay is applied.
    lr_decay : float
        The decay factor that scales the learning rate.
    epoch : int
        The current epoch (or iteration) count. Must be a non-negative integer.

    Returns
    -------
    float
        The adjusted learning rate after applying decay. If lr_decay is zero or negative,
        the function returns the initial learning rate.

    Examples
    --------
    >>> get_decayed_lr(0.1, 0.01, 10)
    0.09090909090909091
    """
    return initial_lr / (1 + lr_decay * epoch) if lr_decay > 0 else initial_lr


def log_progress(epoch, n_iterations, loss, current_lr, verbose, interval=100):
    """
    Print training progress information if verbose logging is enabled.

    Parameters
    ----------
    epoch : int
        The current epoch index (0-indexed).
    n_iterations : int
        Total number of training iterations.
    loss : float
        The loss value computed for the current epoch.
    current_lr : float
        The learning rate being used in the current epoch.
    verbose : bool
        A flag indicating whether to print log messages.
    interval : int, optional
        The interval (in epochs) at which progress should be reported. Default is 100.

    Returns
    -------
    None

    Notes
    -----
    Progress is printed if `verbose` is True and if the current epoch is a multiple of
    `interval` or if it is the last epoch in the training loop.

    Examples
    --------
    >>> log_progress(0, 1000, 0.25, 0.1, True, interval=100)
    Epoch 1/1000: Loss = 0.250000 | LR = 0.100000
    """
    if verbose and (epoch % interval == 0 or epoch == n_iterations - 1):
        print(
            f"Epoch {epoch+1}/{n_iterations}: Loss = {loss:.6f} | LR = {current_lr:.6f}"
        )


def check_early_stopping(
    loss, best_loss, tol, counter, n_iter_no_change, verbose=False, epoch=None
):
    """
    Evaluate and update the early stopping condition.

    Parameters
    ----------
    loss : float
        The current loss value for the epoch.
    best_loss : float
        The best (lowest) loss value recorded so far.
    tol : float
        The minimum improvement in loss required to reset the no-improvement counter.
    counter : int
        The current count of consecutive epochs with no significant improvement.
    n_iter_no_change : int
        The maximum allowed epochs with no improvement before stopping.
    verbose : bool, optional
        If True, prints a message when early stopping is triggered. Default is False.
    epoch : int, optional
        The current epoch index (used for logging purposes). Default is None.

    Returns
    -------
    tuple of (float, int, bool)
        A tuple containing:
            new_best_loss (float): Updated best loss value.
            new_counter (int): Reset or incremented counter.
            stop (bool): True if early stopping condition is met, otherwise False.

    Examples
    --------
    >>> best_loss, counter, stop = check_early_stopping(0.5, 0.6, 0.01, 0, 5, True, 10)
    >>> best_loss, counter, stop
    (0.5, 0, False)
    """
    if loss < best_loss - tol:
        return loss, 0, False  # improvement detected, so reset counter
    else:
        counter += 1
        if counter >= n_iter_no_change:
            if verbose and epoch is not None:
                print(
                    f"Early stopping at epoch {epoch+1}: No improvement for {n_iter_no_change} iterations | Loss = {loss:.6f}"
                )
            return best_loss, counter, True  # signal to stop training
        return best_loss, counter, False
