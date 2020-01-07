from tqdm import tqdm_notebook as tqdm

from scipy import optimize
from scipy.linalg import norm

class ACRNM:
    def __init__(self,
                 inner_method='Nelder-Mead',
                 m=2,
                 n=12,
                 norm_ord=2,
                 max_iter=500,
                 inner_tol=10e-9,
                 tol=1e-6):
        self.inner_method = inner_method
        self.n = n
        self.m = m
        self.norm_ord = norm_ord
        self.max_iter = max_iter
        self.inner_tol = inner_tol
        self.tol = tol
        
        self.x_list = []
        
        
    def _quadratic_aux(self, func, grad, hess, x, y):
        return func(x) + grad(x).T.dot(y - x) + hess(x).dot(y - x).T.dot(y - x)/2
    
    def _cubic_reg(self, func, grad, hess, x, y, L3):
        return self._quadratic_aux(func, grad, hess, x, y) + (self.m*L3/6)*norm(y - x, ord=self.norm_ord)**3
    
    def _psi(self, func, grad, x, L3, k):
        psi_1 = func(self.x_list[1]) + (self.n*L3/6)*norm(x - self.x_list[0], ord=self.norm_ord)**3
        grad_sum = 0
        for i in range(1, k):
            grad_sum += ((i + 1)*(i + 2)/2)*(func(self.x_list[i + 1]) + grad(self.x_list[i + 1]).dot(x - self.x_list[i + 1]))
        return psi_1 + grad_sum
    
    def minimize(self, func, grad, hess, L3, x_init, iterations=None):
        self.x_list = []
        
        if iterations is None:
            iterations = self.max_iter
        
        self.x_list.append(x_init)
        x_1 = optimize.minimize(lambda y: self._cubic_reg(func, grad, hess, x_init, y, L3),
                                x_init,
                                method=self.inner_method,
                                tol=self.inner_tol)
        self.x_list.append(x_1.x)
        
        for i in tqdm(range(1, iterations)):
            if norm(self.x_list[-1] - self.x_list[-2], ord=2) < self.tol:
                print('Early stopped after {} iterations.'.format(i))
                print('Tolerance achieved.')
                break
            v_k = optimize.minimize(lambda x: self._psi(func, grad, x, L3, i),
                                    x_init,
                                    method=self.inner_method,
                                    tol=self.inner_tol)
            y_k = i*self.x_list[i]/(i + 3) + 3*v_k.x/(i + 3)
            x_next = optimize.minimize(lambda y: self._cubic_reg(func, grad, hess, y_k, y, L3),
                                       x_init,
                                       method=self.inner_method,
                                       tol=self.inner_tol)
            self.x_list.append(x_next.x)
            
        return self.x_list[-1]