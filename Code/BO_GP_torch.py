import torch

class GaussProcBayesOpt :
    def __init__(self,kernel_fn,acquisition_fn,solver):
        self.acquisition_fn, self.solver, self.kernel_fn = acquisition_fn, solver, kernel_fn

    def __call__(self, black_box_fn, X_0 : torch.Tensor, iterations : int, opt_type : str = 'MAX'):
        if not(opt_type == 'MAX' or opt_type == 'MIN') :
            return
        x_best = torch.tensor(0)
        y_best = torch.tensor(-torch.inf if opt_type == 'MAX' else torch.inf)
        n = X_0.size(1)
        X = X_0.copy().detach() # type: ignore
        Y = torch.tensor(black_box_fn(X_0))
        Sigma = torch.tensor(n,n) # type: ignore
        for i in range(n) :
            for j in range(n) :
                Sigma[i][j] = self.kernel_fn(X[i],X[j])
            if (opt_type == 'MAX' and Y[i] > y_best) or (opt_type == 'MIN' and Y[i] < y_best) :
                x_best, y_best = X[i], Y[i]
        for _ in range(1,iterations) :
            x_i = self.solver(lambda x : self.acquisition_fn(mu(x,X,Y,Sigma,self.kernel_fn), sigma(x,X,Sigma,self.kernel_fn),y_best), opt_type)
            y_i = black_box_fn(x_i)
            Sigma.reshape([n + 1, n + 1])
            Sigma[n][n] = self.kernel_fn(x_i,x_i)
            Sigma[ [torch.tensor([i, n]) for i in range(n)] ] =  [ self.kernel_fn(x,x_i) for x in X ] # type: ignore
            Sigma[ [torch.tensor([n, i]) for i in range(n)] ] =  [ self.kernel_fn(x_i,x) for x in X ] # type: ignore
            X.reshape(n + 1)
            Y.reshape(n + 1)
            X[n], Y[n] = x_i, y_i
            n += 1
            if (opt_type == 'MAX' and y_i > y_best) or (opt_type == 'MIN' and y_i < y_best) :
                x_best, y_best = x_i, y_i
        return x_best, y_best

def gaussPHI(y_best : torch.Tensor, mu  : torch.Tensor, sigma  : torch.Tensor) :
    return -(torch.erf((y_best - mu) / (torch.sqrt(2) * sigma)) + 1) / 2 # type: ignore
    
def gaussPhi(y_best : float, mu : torch.Tensor, sigma : torch.Tensor) :
    return torch.exp(-torch.square(y_best - mu) / (2 * torch.square(sigma))) / (torch.sqrt(2 * torch.pi) * sigma) # type: ignore

def mu(x : torch.Tensor, X : torch.Tensor, Y : torch.tensor, Sigma : torch.Tensor, kernel_fn) : # type: ignore
    return kernel_fn(X,x).unsqueeze(-2).matmul( Sigma.inverse() ).matmul( Y.unsqueeze(-1) )

def sigma(x : torch.Tensor, X : torch.Tensor, Sigma : torch.Tensor, kernel_fn) :
    return kernel_fn(x,x) - kernel_fn(X,x).unsqueeze(-2).matmul( Sigma.inverse() ).matmul( kernel_fn(X,x).unsqueeze(-1) )

def acquisitionPI(mu : torch.Tensor, sigma : torch.Tensor, y_best : torch.Tensor) :
    return gaussPHI(y_best,mu,sigma)

def acquisitionEI(mu : torch.Tensor, sigma : torch.Tensor, y_best : torch.Tensor) :
    return -( sigma * gaussPhi(y_best,mu,sigma) + (y_best - mu) * gaussPHI(y_best,mu,sigma) ) # type: ignore

def acquisitionLCB(mu : torch.Tensor, sigma : torch.Tensor, kappa : torch.Tensor) :
    return mu - kappa * sigma

def jMu(x : torch.Tensor, X : torch.Tensor, Y : torch.tensor, Sigma : torch.Tensor, kernel_fn, J_kernel_fn) : # type: ignore
    return 2 * J_kernel_fn(X,x).unsqueeze(-2).matmul( Sigma.inverse() ).matmul( Y.unsqueeze(-1) )

def jSigma(x : torch.Tensor, X : torch.Tensor, Sigma : torch.Tensor, kernel_fn, J_kernel_fn) :
    return 2 * ( J_kernel_fn(x,x) - kernel_fn(X,x).unsqueeze(-2).matmul( Sigma.inverse() ).matmul( J_kernel_fn(X,x).unsqueeze(-1) ) )

def jGaussPHI(y_best : torch.Tensor, mu  : torch.Tensor, sigma  : torch.Tensor, J_mu : torch.Tensor, J_sigma  : torch.Tensor) :
    return torch.exp(-torch.square((y_best - mu) / (torch.sqrt(2) * sigma))) * ( J_mu / sigma + (y_best - mu) * J_sigma / torch.square(sigma) ) / torch.sqrt(2 * torch.pi) # type: ignore

def jGaussPhi(y_best : torch.Tensor, mu  : torch.Tensor, sigma  : torch.Tensor, J_mu : torch.Tensor, J_sigma : torch.Tensor) :
    return ( torch.exp( -torch.square(y_best - mu) / ( 2 * torch.square(sigma) ) ) * sigma * ( (y_best - mu) * ( J_mu * sigma + (y_best - mu) * J_sigma ) / ( 2 * torch.pow(sigma,3) ) ) - torch.exp( -torch.square(y_best - mu) / ( 2 * torch.square(sigma) ) ) * J_sigma ) / ( torch.sqrt(2 * torch.pi) * torch.square(sigma) ) # type: ignore

def jAcquisitionPI(y_best : torch.Tensor, J_mu : torch.Tensor, J_sigma : torch.Tensor) :
    return jGaussPHI(y_best,mu,sigma,J_mu,J_sigma) # type: ignore

def jAcquisitionEI(x : torch.Tensor, y_best : torch.Tensor) :
    return -( jSigma * gaussPhi(y_best,mu,sigma) + sigma * jGaussPhi(y_best,mu,sigma,jMu,jSigma) - jMu * gaussPHI(y_best,mu,sigma) + (y_best - mu) * jGaussPHI(y_best,mu,sigma,jMu,jSigma) ) # type: ignore

def jAcquisitionLCB(kappa : torch.Tensor, J_mu : torch.Tensor, J_sigma : torch.Tensor) :
    return J_mu - kappa * J_sigma

def whiteNoiseK(a : torch.Tensor, b : torch.Tensor, sigma : torch.Tensor) :
    return torch.square(sigma)

def gaussK(a : torch.Tensor, b : torch.Tensor, sigma : torch.Tensor, length : torch.Tensor) :
    d = torch.cdist(a.unsqueeze(-2), b.unsqueeze(-2), 2).squeeze(-1).squeeze(-1)
    return torch.square(sigma) * torch.exp( -torch.square(d) / ( 2 * torch.square(length) ) )

def quadK(a : torch.Tensor, b : torch.Tensor, sigma : torch.Tensor, length : torch.Tensor, scale : torch.Tensor) :
    d = torch.cdist(a.unsqueeze(-2), b.unsqueeze(-2), 2).squeeze(-1).squeeze(-1)
    return torch.square(sigma) * torch.pow(1 + torch.square(d) / ( 2 * scale * torch.square(length) ), -scale)

def periodK(a : torch.Tensor, b : torch.Tensor, sigma : torch.Tensor, length : torch.Tensor, period : torch.Tensor) :
    d = torch.cdist(a.unsqueeze(-2), b.unsqueeze(-2), 2).squeeze(-1).squeeze(-1)
    return torch.square(sigma) * torch.exp( -2 / torch.square(length) * torch.square(torch.sin(torch.pi / period * d )) )

def localPeriodK(a : torch.Tensor, b : torch.Tensor, sigma : torch.Tensor, length : torch.Tensor, period : torch.Tensor, scale : torch.Tensor) :
    return periodK(a,b,sigma,length,period) * gaussK(a,b,sigma,scale)

def newtonMethod(iterations : int, x_0 : torch.Tensor, f, J_f) :
    x = x_0
    for _ in range(iterations) :
        x = x - J_f(x).inverse().matmul(f(x).unsqueeze(-1)).squeeze(-1)
    return x



