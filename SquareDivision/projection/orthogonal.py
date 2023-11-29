import numpy as np

def orth_proj_onto_affine_L(x:np.ndarray , A:np.ndarray, b:np.ndarray ):
    """ Return the orthogonal projection of x onto affine subspace L,
            L = {z : A.z == b}.
        Arguments:
            x : (n,)
            A : (k, n) of full rank
            b : (k,)
        Comment:
            Here '.' is matrix multiplication.
            for lambdas = (A.AT)^-1.(-A.x + b) the point
            pt = x + AT.lambdas belong to subspace L, indeed:
                A.pt = A(x + AT.(A.AT)^-1.(-A.x + b))=
                    A.x + A.AT.(A.AT)^-1.(-A.x) + A.AT.(A.AT)^-1.b = 
                    A.x + I.(-A.x) + I.b =
                    A.x-A.x + b =
                    = b 
            A.AT is invertable beacuse A has full rank.
        """
    inv_A_AT = np.linalg.inv(np.matmul(A, A.T))
    lambdas = np.matmul(inv_A_AT, -A.dot(x)+b)
    return x + A.T.dot(lambdas)
# maybe use QR decomposition or least squares ?