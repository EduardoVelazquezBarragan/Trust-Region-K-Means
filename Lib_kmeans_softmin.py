import numpy as np


def _logsumexp_estable(A, axis=1):
    """
    Calcula log(sum(exp(A))) de forma numéricamente estable.
    """
    A_max = np.max(A, axis=axis, keepdims=True)
    return np.squeeze(
        A_max + np.log(np.sum(np.exp(A - A_max), axis=axis, keepdims=True)),
        axis=axis
    )


def distancias_cuadradas(X, W):
    """
    Calcula la matriz de distancias cuadradas entre datos y centros.

    Parámetros
    ----------
    X : ndarray, shape (n, d)
        Datos.
    W : ndarray, shape (k, d)
        Centros.

    Regresa
    -------
    D2 : ndarray, shape (n, k)
        D2[i, j] = ||x_i - w_j||^2
    """
    X = np.asarray(X, dtype=float)
    W = np.asarray(W, dtype=float)

    return np.sum((X[:, None, :] - W[None, :, :]) ** 2, axis=2)


def crear_objetivo_kmeans_softmin(X, k, tau, promedio=False, constante=True):
    """
    Crea la función objetivo suavizada de k-means usando soft-min log-sum-exp.

    El objetivo clásico es:

        F(W) = sum_i min_j 1/2 ||x_i - w_j||^2

    La versión suavizada es:

        F_tau(W) =
            - tau * sum_i log(
                sum_j exp( - (1/2 ||x_i - w_j||^2) / tau )
            )

    Esta función devuelve una función f(z), donde z es el vector de centros
    aplanado, compatible con mRC_SR1(f, x0, ...).

    Parámetros
    ----------
    X : ndarray, shape (n, d)
        Datos.
    k : int
        Número de clusters.
    tau : float
        Parámetro de suavización. Debe ser positivo.
        Si tau es pequeño, se aproxima más al k-means duro.
    promedio : bool
        Si True, divide el objetivo entre n.
        Esto no cambia el minimizador, pero puede mejorar la escala numérica.
    constante : bool
        Si True, suma tau * log(k) por cada dato.
        Esta constante no cambia el minimizador ni el gradiente, pero ayuda
        a que el valor del objetivo no sea tan negativo.

    Regresa
    -------
    f : callable
        Función f(z) -> escalar.
    """
    X = np.asarray(X, dtype=float)

    if X.ndim != 2:
        raise ValueError("X debe tener shape (n, d).")

    if k <= 0:
        raise ValueError("k debe ser positivo.")

    if tau <= 0:
        raise ValueError("tau debe ser positivo.")

    n, d = X.shape

    def f(z):
        z = np.asarray(z, dtype=float)

        if z.size != k * d:
            raise ValueError(
                f"z debe tener tamaño k*d = {k*d}, pero tiene tamaño {z.size}."
            )

        W = z.reshape(k, d)

        # D2[i, j] = ||x_i - w_j||^2
        D2 = distancias_cuadradas(X, W)

        # d_ij = 1/2 ||x_i - w_j||^2
        D = 0.5 * D2

        # A[i, j] = -d_ij / tau
        A = -D / tau

        # softmin_i = -tau log(sum_j exp(-d_ij/tau))
        valores_softmin = -tau * _logsumexp_estable(A, axis=1)

        valor = np.sum(valores_softmin)

        # Constante opcional:
        # softmin_tau(d) + tau log(k)
        # No cambia el minimizador.
        if constante:
            valor += n * tau * np.log(k)

        if promedio:
            valor /= n

        return float(valor)

    return f


def asignaciones_duras(X, W):
    """
    Asigna cada punto al centro más cercano.

    Parámetros
    ----------
    X : ndarray, shape (n, d)
    W : ndarray, shape (k, d)

    Regresa
    -------
    labels : ndarray, shape (n,)
        labels[i] es el cluster asignado al punto x_i.
    """
    D2 = distancias_cuadradas(X, W)
    return np.argmin(D2, axis=1)


def responsabilidades_suaves(X, W, tau):
    """
    Calcula responsabilidades suaves tipo soft k-means.

    r_ij =
        exp(-||x_i - w_j||^2 / (2 tau))
        /
        sum_l exp(-||x_i - w_l||^2 / (2 tau))

    Parámetros
    ----------
    X : ndarray, shape (n, d)
    W : ndarray, shape (k, d)
    tau : float

    Regresa
    -------
    R : ndarray, shape (n, k)
        Matriz de responsabilidades suaves.
    """
    if tau <= 0:
        raise ValueError("tau debe ser positivo.")

    D2 = distancias_cuadradas(X, W)
    A = -0.5 * D2 / tau

    A = A - np.max(A, axis=1, keepdims=True)
    R = np.exp(A)
    R = R / np.sum(R, axis=1, keepdims=True)

    return R