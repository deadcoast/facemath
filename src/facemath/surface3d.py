from __future__ import annotations
import numpy as np
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import eigsh


def vertex_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    v = np.asarray(vertices, float); f = np.asarray(faces, int)
    n = np.zeros_like(v)
    tri = v[f]
    fn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    for k in range(3):
        np.add.at(n, f[:, k], fn)
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    return n


def cotangent_laplacian(vertices: np.ndarray, faces: np.ndarray):
    v = np.asarray(vertices, float); f = np.asarray(faces, int)
    rows=[]; cols=[]; data=[]
    area = np.zeros(len(v))
    for tri in f:
        i,j,k = tri
        vi,vj,vk = v[i],v[j],v[k]
        e0, e1, e2 = vj-vk, vk-vi, vi-vj
        def cot(a,b):
            return float((a @ b) / max(np.linalg.norm(np.cross(a,b)), 1e-12))
        coti = cot(vj-vi, vk-vi)
        cotj = cot(vi-vj, vk-vj)
        cotk = cot(vi-vk, vj-vk)
        for a,b,w in [(j,k,coti),(i,k,cotj),(i,j,cotk)]:
            rows += [a,b]; cols += [b,a]; data += [w/2,w/2]
        ar = np.linalg.norm(np.cross(vj-vi, vk-vi))/2
        area[tri] += ar/3
    w = coo_matrix((data,(rows,cols)), shape=(len(v),len(v))).tocsr()
    d = np.asarray(w.sum(1)).ravel()
    l = diags(d) - w
    m = diags(np.maximum(area,1e-12))
    return l, m


def laplace_beltrami_eigensystem(vertices: np.ndarray, faces: np.ndarray, k: int = 32):
    l,m = cotangent_laplacian(vertices,faces)
    k = min(k, len(vertices)-2)
    vals, vecs = eigsh(l, k=k, M=m, sigma=0.0, which='LM')
    order=np.argsort(vals)
    return vals[order], vecs[:,order]


def heat_kernel_signature(vertices: np.ndarray, faces: np.ndarray, times: np.ndarray, k: int = 32) -> np.ndarray:
    vals, vecs = laplace_beltrami_eigensystem(vertices,faces,k)
    return np.stack([np.sum((vecs**2)*np.exp(-vals[None,:]*t),axis=1) for t in times],axis=1)


def wave_kernel_signature(vertices: np.ndarray, faces: np.ndarray, energies: np.ndarray, sigma: float = 1.0, k: int = 32) -> np.ndarray:
    vals, vecs = laplace_beltrami_eigensystem(vertices,faces,k)
    loge = np.log(np.maximum(vals,1e-12))
    out=[]
    for e in energies:
        w=np.exp(-((e-loge)**2)/(2*sigma**2))
        w/=max(w.sum(),1e-12)
        out.append(np.sum((vecs**2)*w[None,:],axis=1))
    return np.stack(out,axis=1)


def spectral_shape_distance(vertices_a, faces_a, vertices_b, faces_b, k: int = 16) -> float:
    va,_=laplace_beltrami_eigensystem(vertices_a,faces_a,k)
    vb,_=laplace_beltrami_eigensystem(vertices_b,faces_b,k)
    m=min(len(va),len(vb))
    # scale-normalize spectrum by first non-zero eigenvalue
    va=va[:m]/max(va[1] if m>1 else 1,1e-12)
    vb=vb[:m]/max(vb[1] if m>1 else 1,1e-12)
    return float(np.linalg.norm(va-vb)/np.sqrt(m))


def functional_map_from_descriptors(phi_a: np.ndarray, phi_b: np.ndarray, desc_a: np.ndarray, desc_b: np.ndarray, ridge: float = 1e-6) -> np.ndarray:
    a = phi_a.T @ desc_a
    b = phi_b.T @ desc_b
    return b @ a.T @ np.linalg.inv(a @ a.T + ridge * np.eye(a.shape[0]))
