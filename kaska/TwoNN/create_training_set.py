import scipy.stats as ss
from lhd import lhd
import numpy as np

def create_training_set(parameters, minvals, maxvals, fix_params=None, n_train=200):
    """Creates a traning set for a set of parameters specified by
    ``parameters`` (not actually used, but useful for debugging
    maybe). Parameters are assumed to be uniformly distributed
    between ``minvals`` and ``maxvals``. ``n_train`` input parameter
    sets will be produced, and returned with the actual distributions
    list. The latter is useful to create validation sets.
    It is often useful to add extra samples for regions which need to
    be carefully evaluated. We do this by adding a `fix_params` parameter
    which should be a dictionary indexing the parameter name, its fixed
    value, and the number of additional samples that will be drawn.
    Parameters
    -------------
    parameters: list
        A list of parameter names
    minvals: list
        The minimum value of the parameters. Same order as ``parameters``
    maxvals: list
        The maximum value of the parameters. Same order as ``parameters``
    fix_params: dictionary
        A diciontary indexed by the parameter name. Each item will have a
        tuple indicating the fixed value of the parameter, and how many
        extra LHS samples are required
    n_train: int
        How many training points to produce
    Returns
    ---------
    The training set and a distributions object that can be used by
    ``create_validation_set``
    """

    distributions = []
    for i, p in enumerate(parameters):
        distributions.append(
            ss.uniform(loc=minvals[i], scale=(maxvals[i] - minvals[i]))
        )
    samples = lhd(dist=distributions, size=n_train)

    if fix_params is not None:
        # Extra samples required
        for k, v in list(fix_params.items()):
            # Check whether they key makes sense
            if k not in parameters:
                raise ValueError(
                    "You have specified '%s', which is" % k
                    + " not in the parameters list"
                )

            extras = fix_parameter_training_set(
                parameters, minvals, maxvals, k, v[0], v[1]
            )
            samples = np.r_[samples, extras]

    return samples, distributions


def create_validation_set(distributions, n_validate=500):
    """Creates a validation set of ``n_validate`` vectors, using the
    ``distributions`` list.
    Parameters
    ------------
    distributions: list
        A set of parameter distributions (np.random objects)
    n_validate: int (optional)
        The number of samples to draw from `distributions`.
    Returns
    -------
    A bunch of samples
    """
    validate = []
    for d in distributions:
        validate.append(d.rvs(n_validate))
    validate = np.array(validate).T
    return validate
