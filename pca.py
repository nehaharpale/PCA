import numpy as np

#X = np.array([[-1, -1],[-1, 1], [1, -1], [1, 1]]) 
#total_samples = X.shape[0]

def compute_Z(X, centering, scaling):
    if centering == True:
        Z = X - np.mean(X, axis=0, keepdims=True)
    if scaling == True:
        Z = Z / Z.std(0)
    if centering == False & scaling==False:
        Z=X
    return Z

def compute_covariance_matrix(Z):
    COV = (Z.T).dot(Z)
    return COV

def find_pcs(COV):
    L,PCS = np.linalg.eig(COV)
    idx = L.argsort()[::-1]
    L[idx].sort()
    PCS = PCS[:,idx]
    return L,PCS

def project_data(Z, PCS, L, k, var):
    count = 0
    lsum = 0.0
    num = 0
    if var >0 and var <= 1:
        pass
    else:
        if var == 0:
            var = k
        else:
            print("Value out of range ")
    if k==0:
            for i in L[0:-1]:
                while count==0:
                    final = 0.0
                    lsum = lsum + L[num]
                    final = lsum/np.sum(L)
                    num = num+1
                    if final >= var:
                        k = num
                        count = 1


    if k > 0 and k <= Z.shape[1]:
            pass


    PCS = PCS[:,:k]
    Z_star = Z.dot(PCS)
    return Z_star

#Z = compute_Z(X, centering = True, scaling = False)
#COV = compute_covariance_matrix(Z)
#L,PCS = find_pcs(COV)
#k =0
#var =1
#Z_star = project_data(Z, PCS, L, k, var)
