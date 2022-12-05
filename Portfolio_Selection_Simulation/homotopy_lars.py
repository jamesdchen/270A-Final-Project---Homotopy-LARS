import numpy as np

def homotopy_LARS(A, y):
    # LARS algo for LASSO path (aka homotopy LARS for LASSO,
    # in constrast to forward variable selection LARS)
    m, n = A.shape

    # start with 0 model
    x = np.zeros((n, 1))

    # calculate correlation of variables with responses
    # to determine variable with largest correlation
    c0 = A.T @ y
    
    # equicorrelation variables set
    I = [np.argmax(np.abs(c0))]

    # equicorrelation signs set
    s = np.zeros_like(x)
    s[I] = np.sign(c0[I])

    # calculate emperical covariance matrix
    ATA = A.T @ A

    # # see eqn (5) of "The Lasso Problem and Uniqueness" (TLPU)
    # s[I] = np.sign(c0 - ATA @ x[I])

    # linear factors as defined in eqn (15) of TLPU
    # needs to be pre-allocated because crossing time
    # calculation depends on indexing a full vector
    # as opposed to a variable length set
    c = np.zeros_like(x)
    d = np.zeros_like(x)

    # set regularization parameter lambda = infty
    lambda_ = np.abs(c0).max()

    # initialize iterate
    # k = 0

    # initialize solution list
    xs = []

    # initialize list to hold LASSO parameter path
    lambdas = [lambda_]

    # create CONST set of integers from 1 to n
    i_set = set(range(n))

    while lambda_ > .0005:
    # while lambda_ > .0005:

        ## step 1: compute LARS lasso solution at lambda_k by least squares
        ## (i.e. KKT stationarity condition for lasso)

        # update model at equicorrelated variables as seen in eqn (15) of TLPU
        c[I] = np.linalg.pinv(A[:,I]) @ y
        d[I] = np.linalg.pinv(A[:,I].T @ A[:,I]) @ s[I]

        # update model at equicorrelated variables as seen in eqn (15) of TLPU
        x = c - lambda_*d

        ## step 2: compute next joining time
        # needs to be transposed because weird matrix calculation thing
        t_join_p = (ATA[:,I] @ c[I] - c0)/(ATA[:,I] @ d[I] + 1)
        t_join_m = (ATA[:,I] @ c[I] - c0)/(ATA[:,I] @ d[I] - 1)
        # print(t_join_m)

        # vectorized calculation of t_join (multiplication gives AND)
        t_join = np.where((np.zeros_like(t_join_p) <= t_join_p)   * (t_join_p <= lambda_*np.ones_like(t_join_p)),\
                          t_join_p, t_join_m)
        # calculate inactive set
        Ic = list(i_set - set(I))

        # calculate next joining time (eqn (17) from TLPU
        # on a typical iteration, the try block will run
        # on the occasion all of the variables have been
        # already been added to the equicorrelation set,
        # we want to directly choose the crossing variables
        try:
          lambda_join = t_join[Ic].max()
          # calculate next joining variable ()
          join_var = [np.argwhere(t_join==lambda_join)[0,0]]

          # calculate next joining sign
          s_join = np.sign(c0[join_var] - A[:,join_var].T @ A @ (c - lambda_join*d))

        except:
          # as lambda_ is a non-negative scalar,
          # 0 is a guaranteed lower bound
          lambda_join = 0

        ## step 3: compute next crossing time

        # as seen in TLPU in line above eqn (19)
        lambda_ratio = c/d

        # vectorized calculation of t_cross (multiplication gives AND)
        # numerical instability and rounding forces us to take epsilon off
        t_cross = np.where((np.zeros_like(lambda_ratio) <= lambda_ratio) *\
                           (lambda_ratio <= (lambda_ - .00005)*np.ones_like(lambda_ratio)),\
                          lambda_ratio, np.zeros_like(lambda_ratio))

        # calculate next crossing time eqn(20) in TLPU
        lambda_cross = t_cross[I].max()

        # calculate next crossing variable
        cross_var = [np.argwhere(t_cross==lambda_cross)[0,0]]

        # calculate next crossing sign
        s_cross = s[cross_var]

        ## step 4: set lambda_{k+1} and adjust equicorrelation set accordingly
        
        if lambda_join > lambda_cross:

            # update lambda w/ lambda_join
            lambda_ = lambda_join
            
            # add joining variable to equicorrelation set
            I.append(join_var[0])
            # I = list(set(I))

            # add joining sign to equicorrelation signs
            s[join_var] = s_join

        else:

            # update lambda w/ lambda_cross
            lambda_ = lambda_cross

            # remove crossing variable from equicorrelation set
            I.remove(cross_var)

            # remove crossing signs from equicorrelation signs
            s[cross_var] = 0

        # append solution x to solution list before updating solution x
        xs.append(x.copy())

        # append regularization parameter to lasso path list before updating lambda
        lambdas.append(lambda_)
        
        # update k = k + 1
        # k += 1

    return np.array(xs).squeeze()