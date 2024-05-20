import numpy as np
import tensorly as tl
from tensorly.decomposition._cp import initialize_cp

from tensorly.tenalg import khatri_rao
from tqdm import tqdm

def cp_als(tensor: np.ndarray, R, max_iter):
    N = tl.ndim(tensor)
    # Step 1
    lbd, A = initialize_cp(tensor, R, init='svd', svd='numpy_svd',
                           random_state=0,
                           normalize_factors=True)
    # A = []
    # for n in range(N):
    #     np.random.seed(N)
    #     A.append(tl.tensor(np.random.random((tensor.shape[n], rank))))
    # lbd = tl.ones(rank)

    for epoch in tqdm(range(max_iter)):
        for n in range(N):
            # Step 2
            V = np.ones((R, R))
            for i in range(N):
                if i != n:
                    V = np.matmul(A[i].T, A[i]) * V
            # V = None
            # for i in range(N):
            #     if i != n:
            #         if V is None:
            #             V = np.matmul(A[i].T, A[i])
            #         else:
            #             V = np.matmul(A[i].T, A[i]) * V


            # Step 3
            T = khatri_rao(A, skip_matrix=n)
            A[n] = np.matmul(np.matmul(tl.unfold(tensor, mode=n), T), np.linalg.pinv(V))

            # Step 4
            for r in range(R):
                lbd[r] = tl.norm(A[n][:, r])
            A[n] = A[n] / tl.reshape(lbd, (1, -1))
		# Step 5
        tensor_pred = tl.fold(np.matmul(np.matmul(A[0], np.diag(lbd)),
                                        khatri_rao(A, skip_matrix=0).T),
                              mode=0,
                              shape=tensor.shape)
        if tl.norm(tensor - tensor_pred) <= 1e-7:
            return A, lbd, epoch

    return A, lbd, max_iter

if __name__ == '__main__':
    np.random.seed(10086)
    inpt = tl.tensor(np.random.random((3, 3, 3)), dtype=np.float32)
    print(inpt.shape)
    A, lbd, epoch = cp_als(inpt, R=5, max_iter=1000)
    print(A[0].shape)
    print(A[1].shape)
    tensor_pred = tl.fold(np.matmul(np.matmul(A[0], np.diag(lbd)),
                                    khatri_rao(A, skip_matrix=0).T),
                          mode=0,
                          shape=inpt.shape)

    print(tl.norm(inpt - tensor_pred), epoch)
    print(tensor_pred.shape)