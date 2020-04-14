from utils import *

def mean_and_gaussian_curvatures(Ix, Iy, Ixx, Iyy, Ixy):

	H = np.zeros((Ix.shape[0], Ix.shape[1]))
	K = np.zeros((Ix.shape[0], Ix.shape[1]))

	for i in range(Ix.shape[0]):
		for j in range(Ix.shape[1]):
			E = 1+Ix[i][j]**2
			F = Ix[i][j]*Iy[i][j]
			G = 1+Iy[i][j]**2
			den = np.sqrt(1+Ix[i][j]**2+Iy[i][j]**2)
			e = -Ixx[i][j]/den
			f = -Ixy[i][j]/den
			g = -Iyy[i][j]/den
			den = E*G-F**2
			H[i][j] = (E*g+G*e-2*F*f)/(2*den)
			K[i][j] = (e*g-f**2)/den

	return H, K

def principal_curvatures(H, K):

	K1 = np.zeros((H.shape[0], H.shape[1]))
	K2 = np.zeros((H.shape[0], H.shape[1]))

	for i in range(K1.shape[0]):
		for j in range(K1.shape[1]):
			sq = H[i][j]**2-K[i][j]
			if abs(sq)<=1e-14:
				sq=0.0
			K1[i][j] = H[i][j] + np.sqrt(sq)
			K2[i][j] = H[i][j] - np.sqrt(sq)

	return K1, K2

def topology_HK(H, K):

	Kminus = {-1:'peak', 0:'none', 1:'pit'}
	Kzero = {-1:'ridge', 0:'flat', 1:'valley'}
	Kplus = {-1:'saddle ridge', 0:'minimal surface', 1:'saddle valley'}
	curv_dict_KH = {-1:Kminus, 0:Kzero, 1:Kplus}

	return curv_dict_KH[K][H]

def topology_K2K1(K2, K1):

	K1minus = {-1:'peak', 0:'ridge', 1:'saddle'}
	K1zero = {-1:'ridge', 0:'flat', 1:'valley'}
	K1plus = {-1:'saddle', 0:'valley', 1:'pit'}
	curv_dict_K1K2 = {-1:K1minus, 0:K1zero, 1:K1plus}

	return curv_dict_K1K2[K1][K2]

def find_NPS(img, threshold):

	for i in range(img.shape[0]):
		for j in range(img.shape[1]):
			print("[{}][{}] : {}".format(i, j, NPS_pixel(img, (i, j), threshold)))

	
