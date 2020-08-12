import numpy as np

# Pearson Correlation Coefficient, [-1, 1], 1 is best
def PCC(xs, ys):
    D = np.array([xs, ys])
    assert(D.ndim == 2)
    # 2x2
    RR = np.corrcoef(D)
    return RR[0, 1]

# Mean Absolute Error, >=0, 0 is best
def MAE(xs, ys):
    diff = np.array(xs) - np.array(ys)
    assert(diff.ndim == 1)
    val = np.sum(np.abs(diff)) / diff.shape[0]
    return val

def ICC(xs, ys, icc_type = 'icc3'):
    Y = np.array([xs, ys]).T
    return run_ICC(Y, icc_type)


# Intra-Class Correlation, [0,1], 1 is best 
def run_ICC(Y, icc_type = 'icc3'):
        # Y: size n * 2(k)
        #Y = np.array([xs, ys]).T
        n, k = Y.shape

        # Degrees of Freedom
        dfc = k - 1
        dfe = (n - 1) * (k - 1)
        dfr = n - 1

        # Sum Square Total
        mean_Y = np.mean(Y)
        SST = ((Y - mean_Y) ** 2).sum()

        # create the design matrix for the different levels
        x = np.kron(np.eye(k), np.ones((n, 1)))  # sessions
        x0 = np.tile(np.eye(n), (k, 1))  # subjects
        X = np.hstack([x, x0])

        # Sum Square Error
        predicted_Y = np.dot(np.dot(np.dot(X, np.linalg.pinv(np.dot(X.T, X))), X.T), Y.flatten('F'))
        residuals = Y.flatten('F') - predicted_Y
        SSE = (residuals ** 2).sum()

        MSE = SSE / dfe

        # Sum square column effect - between colums
        SSC = ((np.mean(Y, 0) - mean_Y) ** 2).sum() * n
        MSC = SSC / dfc / n

        # Sum Square subject effect - between rows/subjects
        SSR = SST - SSC - SSE
        MSR = SSR / dfr

        if icc_type == 'icc2':
            # ICC(2,1) = (mean square subject - mean square error) /
            # (mean square subject + (k-1)*mean square error +
            # k*(mean square columns - mean square error)/n)
            ICC = (MSR - MSE) / (MSR + (k-1) * MSE + k * (MSC - MSE) / n)

        elif icc_type == 'icc3':
            # ICC(3,1) = (mean square subject - mean square error) /
            # (mean square subject + (k-1)*mean square error)
            ICC = (MSR - MSE) / (MSR + (k-1) * MSE)
        else:
            raise NotImplementedError("This method isn't implemented yet.")

        return ICC

def test():
    xs = [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8]
    ys = [1, 2, 0, 1, 1, 3, 3, 2, 3, 8, 1, 4, 6, 4, 3, 3, 6, 5, 5, 6, 7, 5, 6, 2, 8, 7, 7, 9, 9, 9, 9, 8]
    print("PCC:", PCC(xs, ys))
    print("MAE:", MAE(xs, ys))
    print("ICC:", ICC(xs, ys))

if __name__ == "__main__":
    test()
