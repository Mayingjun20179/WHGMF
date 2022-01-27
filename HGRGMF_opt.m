function P = HGRGMF_opt(W,train_mat,A_sim,B_sim,option)
% k: Number of neighbors
% max_iter: Maximum number of iteration
% lr: learning rate
% lamb: trade-off parameter for norm-2 regularization on matrix U and V
% K: dimension of the subspace expanded by self-representing vectors(i.e. the dimension for MF)
% beta: trade-off parameter for norm-2 regularization on matrix U and V
% r1: trade-off parameter of graph regularization on nodes in A
% r2: trade-off parameter of graph regularization on nodes in B
% c: constant of the important level for positive sample
% cvs: cross validation setting (1, 2 or 3)

add_eps = 0.01;

self.eps = eps;
self.k = 5;
self.n = 0;
self.K = 50;
self.num_factors = self.K;
self.max_iter = 100;
self.lr = 0.1;
% self.lamb = 0.0333;
self.lamb = option(3);
self.r1 = option(1);
self.r2 = option(2);
self.c = 5;
self.loss = num2cell(inf*ones(1,50));
self.WK = 5;
self.eta = 0.5;

self.imp1 = 1;
self.imp2 = 1;
% W: Mask for training set
%     train_mat: complete interaction matrix
%     A_sim: similarity matrix for nodes in A
%     B_sim: similarity matrix for nodes in B
%     seed: random seed to determine a random state
Y = train_mat .* W;
[self.num_A, self.num_B] = size(Y);

self.n = self.n+1;
self.A_sim = A_sim;
self.B_sim = B_sim;

% emphasize the diag of similarity matrix
self.A_sim = self.A_sim;
self.B_sim = self.B_sim;

% sparsification
[self.A_HL,self.A_clique] = construct_Hypergraphs_knn(self.A_sim,self.k);
[self.B_HL,self.B_clique] = construct_Hypergraphs_knn(self.B_sim,self.k);


% normalization
self.ZA = diag(ones(self.num_A,1)./ sum(self.A_clique + add_eps,2))*(self.A_clique + add_eps);
self.ZB = (self.B_clique +add_eps) *  diag(ones(1,self.num_B)./ sum(self.B_clique + add_eps,1));

% initialization for U and V
seed = 1;
randn('state',seed);
self.U = sqrt(1/self.num_factors)*(randn(self.num_A,self.num_factors));
randn('state',seed);
self.V = sqrt(1/self.num_factors)*(randn(self.num_B,self.num_factors));
[u, s, v] = svd(Assist_method_Utils.WKNKN(Y, A_sim, B_sim, self.WK, self.eta));


self.U(:, 1:min(self.num_factors, min(size(Y)))) = u(:, 1:min(self.num_factors, min(size(Y))));
self.V(:, 1:min(self.num_factors, min(size(Y)))) = v(:, 1:min(self.num_factors, min(size(Y))));
clear u s v;

max_iter = self.max_iter;

% Using adam optimizer:
lr = self.lr;
patient = 3;
numiter = max_iter;
minloss = inf;

% store the initial value of ZA ZB U V for later use
init_U = self.U;
init_V = self.V;
init_ZA = self.ZA;
init_ZB = self.ZB;

% A_old refer to the value of ZA in the last iteratio
self.A_old = self.ZA;
self.B_old = self.ZB;
self.U_old = self.U;
self.V_old = self.V;

ZA_best = zeros(size(self.ZA));
ZB_best = zeros(size(self.ZB));
U_best = zeros(size(self.U));
V_best = zeros(size(self.V));
W_all = W;
% iteration
while numiter > 0
    W_1 = W_all;
    Y_p = self.ZA*self.U*(self.V)'* self.ZB;
    P = sigmoid(Y_p);
    
    %update U,V
    % reinitialize the optimizer
    [self,P] = update_UV(self,W_1,P,Y,max_iter,numiter);

    % store matrix U, V, ZA and ZB for the currently lowest loss for later use
    self.loss{self.n} = [self.loss{self.n},loss_function(self,Y,W_all)];
    if self.loss{self.n}(end) <  minloss
        ZA_best = self.ZA;        ZB_best = self.ZB;
        U_best = self.U;        V_best = self.V;
        minloss = self.loss{self.n}(end);
    end
    
    % if diverge reinitialize U, V, ZA, ZB and optimizer with half the present learning rate
    if self.loss{self.n}(end) > self.loss{self.n}(1) * 2
        if patient==0
            self.ZA = ZA_best;            self.ZB = ZB_best;
            self.U = U_best;            self.V = V_best;
        end
        % Reinitialization
        self.ZA = init_ZA;
        self.ZB = init_ZB;
        self.U = init_U;
        self.V = init_V;
        lr = lr * 0.5;
        self.lr = lr;
        numiter = max_iter;
        self.loss{self.n}(end) = inf;
        patient = patient-1;
        break;
    end

    
    % Update ZA & ZB
    self = update_ZAB(self,P,W_1,Y);
    % store matrix U, V, ZA and ZB for the currently lowest loss
    self.loss{self.n} = [self.loss{self.n},loss_function(self,Y,W_all)];
    if self.loss{self.n}(end) < minloss
        ZA_best = self.ZA;
        ZB_best = self.ZB;
        U_best = self.U;
        V_best = self.V;
        minloss = self.loss{self.n}(end);
    end
    
    % reinitialize U, V, ZA, ZB and optimizer with half the present learning rate if loss diverge
    if self.loss{self.n}(end) > self.loss{self.n}(1) * 2
        if patient ==  0
            self.ZA = ZA_best;
            self.ZB = ZB_best;
            self.U = U_best;
            self.V = V_best;
        end
        % Reinitialization
        self.ZA =  init_ZA;
        self.ZB = init_ZB;
        self.U = init_U;
        self.V = init_V;
        self.lr = self.lr * 0.5;
        numiter = max_iter;
        self.loss{self.n}(end) = inf;
        patient = patient-1;
        break;
    else
        delta_loss = abs(self.loss{self.n}(end) - self.loss{self.n}(end-1)) / abs(self.loss{self.n}(end-1));
        if delta_loss < 1e-4
            numiter = 0;
        end
    end
    numiter = numiter-1;
end

% retrieve the best U, V, ZA and ZB (at lowest loss)
if self.loss{self.n}(end) > minloss
    self.ZA = ZA_best;
    self.ZB = ZB_best;
    self.U = U_best;
    self.V = V_best;
end

Y_p = self.ZA*self.U*(self.V)'*self.ZB;
P = sigmoid(Y_p);

end


function loss = loss_function(self,Y,W)

% Return the value of loss function
% Args:
% Y: interaction matrix
% W: mask for training set
%
% Returns:
% value of loss function

temp = self.ZA*self.U*(self.V)'*self.ZB;
% logexp(temp > 50) = temp(temp > 50) ;  
% logexp(temp <= 50) = log(exp(temp(temp <= 50)) + 1);
logexp =  log(exp(temp) + 1);


%%%%%目标1
f1 = ((1 + self.c * Y - Y).*logexp - self.c * Y.*temp).*W;
f1 = sum(f1(:));

%%%%%目标2
f2 = self.lamb * (norm(self.U,'fro')^2+norm(self.V,'fro')^2);

%%%%%目标3
f3 = self.r1 * trace((self.ZA'*self.U)'*self.A_HL*self.ZA*self.U)+...
    self.r2 * trace((self.ZB'*self.V)'*self.B_HL*self.ZB*self.V);

%%%%%总目标
loss = f1+f2+f3;

end



function output = sigmoid(x)
output =1./(1+exp(-x));
end

function [self,P] = update_UV(self,W_1,P,Y,max_iter,numiter)
%P: predict score matrix
%W_1: Mask for training set
%Y： train interaction matrix

U = gpuArray(single(self.U));  U_old = U;
V = gpuArray(single(self.V));  V_old = V;

 
U_m0 = zeros(size(U),'gpuArray'); U_m0 = single(U_m0);
U_v0 = U_m0;
V_m0 = zeros(size(V),'gpuArray'); V_m0 = single(V_m0);
V_v0 =  V_m0;

ZA = gpuArray(single(self.ZA)); 
ZB = gpuArray(single(self.ZB));

PW_1 = gpuArray(single(P .* W_1)); 

LA_sim = gpuArray(single(self.A_HL));
LB_sim = gpuArray(single(self.B_HL));

Y = gpuArray(single(Y));
for foo = 1:30
    % compute the derivative of U and V
    deriv_U = ZA'*PW_1*(ZB'*V)+...
        (self.c - 1) * ZA'*(Y .* PW_1)*ZB'*V-...
        self.c*(ZA)'*(Y .* W_1)*(ZB)'*V+...
        2 * self.lamb * U+...
        2*self.r1*(ZA)'* LA_sim * ZA * U ;
    
    deriv_V = ZB*PW_1'*ZA*U+...
        (self.c - 1) * ZB*(Y .* PW_1)'*ZA*U-...
        self.c*ZB*(Y .* W_1)'*ZA*U+...
        2 * self.lamb * V+...
        2*self.r2*ZB* LB_sim*(ZB)'*V;
    
    % update using adam optimizer
    [update,U_m0,U_v0] = opter_delta_U(deriv_U,max_iter - numiter,U_m0,U_v0,self.lr);
    U = U + update;
    [update,V_m0,V_v0] = opter_delta_V(deriv_V,max_iter - numiter,V_m0,V_v0,self.lr);
    V = V + update;
    
    Y_p = ZA * U * (V)'* ZB;
    P = sigmoid(Y_p);    
    PW_1 = gpuArray(single(P .* W_1)); 
    % break the loop if reach converge condition
    ob1 = norm(U - U_old, 'fro') / norm(U_old, 'fro')< 0.01;
    ob2 = norm(V - V_old, 'fro')/ norm(V_old, 'fro')< 0.01;
    if  ob1 & ob2
        break;
    end
    U_old = U;
    V_old = V;
end
self.U = gather(U);
self.V = gather(V);
end



function [update,U_m0,U_v0] = opter_delta_U(deriv,iter,U_m0,U_v0,lr)
beta1 = 0.9;
beta2 = 0.9;
epsilon = 10e-8;
t = (iter + 1);
grad = deriv;
m_t = beta1 * U_m0 + (1 - beta1) * grad;
v_t = beta2 * U_v0 + (1 - beta2) * grad .^ 2;
m_cap = m_t / (1 - beta1 ^ t + eps);
v_cap = v_t / (1 - beta2 ^ t + eps);
update = - lr * m_cap ./ (sqrt(v_cap) + epsilon + eps);
U_m0 = m_t;
U_v0 = v_t;
end

function [update,V_m0,V_v0] = opter_delta_V(deriv,iter,V_m0,V_v0,lr)
beta1 = 0.9;
beta2 = 0.9;
epsilon = 10e-8;
t = (iter + 1);
grad = deriv;
m_t = beta1 * V_m0 + (1 - beta1) * grad;
v_t = beta2 * V_v0 + (1 - beta2) * grad .^ 2;
m_cap = m_t / (1 - beta1 ^ t + eps);
v_cap = v_t / (1 - beta2 ^ t + eps);
update = - lr * m_cap ./ (sqrt(v_cap) + epsilon + eps);
V_m0 = m_t;
V_v0 = v_t;
end



function [self,P] = update_ZAB(self,P,W_1,Y)
U = gpuArray(single(self.U));  
V = gpuArray(single(self.V));  
ZA = gpuArray(single(self.ZA));    A_old = ZA;
ZB = gpuArray(single(self.ZB));    B_old = ZB;
A_clique = gpuArray(single(self.A_clique)); 
B_clique = gpuArray(single(self.B_clique));
Y = gpuArray(single(Y));
PW_1 = gpuArray(single(P .* W_1)); %这个表示关于训练集上的预测值

for n = 1:30
    temp_p = ((U*V'*ZB)'+abs((U*V'*ZB)'))*0.5;  %这个是Yjian*(U*V'*ZB)'的正部
    temp_n = (abs((U*V'*ZB)')-(U*V'*ZB)') * 0.5; %这个是Yjian*(U*V'*ZB)'的负部
    UUT_p = (U * U' + abs(U*U')) * 0.5;  %这个是U*U'的正部
    UUT_n = (abs(U * U') - U*U') * 0.5;
    DAN_sim = gpuArray.eye(self.num_A);
    
    D_AP = PW_1 * temp_p + (self.c - 1) * ((Y .* PW_1)*temp_p)+...
        self.c *(Y .* W_1) * temp_n + ...
        2 * self.r1 * DAN_sim * ZA * UUT_p+...
        2 * self.r1 * A_clique * ZA* UUT_n;  %这个表示DA的正部
    
    D_AN = PW_1 * temp_n + (self.c - 1) * (Y .* PW_1) * temp_n+...
        + self.c * (Y .* W_1) * temp_p+...
        2 * self.r1 * DAN_sim * ZA * UUT_n+...
        2 * self.r1 * A_clique * ZA* UUT_p;
    
    temp_p = ((ZA * U * V')'+abs(ZA * U * V')') * 0.5;
    
    temp_n = (abs((ZA * U * V')')-(ZA * U * V')')*0.5;
    
    
    VVT_p = (V * V' + abs(V * V')) * 0.5;    
    VVT_n = (abs(V * V') - V * V') * 0.5;
    DBN_sim = gpuArray.eye(self.num_B);
    D_BP = (temp_p * PW_1 + (self.c - 1) * (temp_p* (Y .* PW_1))+...
        self.c * temp_n * (Y .* W_1)+...        
        2 * self.r2 * VVT_p * ZB * DBN_sim+...
        2 * self.r2 * VVT_n * ZB * B_clique);
    
    D_BN = (temp_n * PW_1 + (self.c - 1) * (temp_n* (Y .* PW_1))+...
        self.c * temp_p * (Y .* W_1)+...
        2 * self.r2 * VVT_n * ZB * DBN_sim+...
        2 * self.r2 * VVT_p * ZB * B_clique);
    
    temp = sum(ZA ./ (D_AP + eps),2); D_SA = diag(temp);
    E_SA = sum(ZA .* D_AN ./ (D_AP + eps),2);
    E_SA = repmat(E_SA,1,self.num_A);
    
    temp = sum(ZB ./ (D_BP + eps),1);D_SB = diag(temp);
    E_SB = sum(ZB .* D_BN ./ (D_BP + eps),1);
    E_SB = repmat(E_SB,self.num_B,1);
    
    ZA = ZA .* (D_SA*D_AN + 1) ./ (D_SA* D_AP + E_SA + eps);
    ZB = ZB .* (D_BN*D_SB + 1) ./ (D_BP* D_SB + E_SB + eps);
    Y_p = ZA*U*V'* ZB;
    P = sigmoid(Y_p);
    PW_1 = gpuArray(single(P .* W_1)); 
    % break the loop if reach convergence condition
    ob3 = norm(ZA - A_old, 'fro') / norm(A_old, 'fro')< 0.01;
    ob4 = norm(ZB - B_old, 'fro')/ norm(B_old, 'fro')< 0.01;
    
    if ob3 & ob4
        break;
    end
    A_old = ZA;
    B_old = ZB;
end
self.ZA = gather(ZA);
self.ZB = gather(ZB);

end


%%%%计算超边权重
%S表示相似性矩阵
%H超图指示矩阵，行表示顶点，列表示边
function We = cal_hyperedge_weight(S,H)
%%%计算距离矩阵
S = (S+S')/2;
D = 1./(exp(S));
D(1:(size(D,1)+1):end) = 0;

[numV,nume] = size(H);  
wol = zeros(1,nume);
for j = 1:nume
    ind = find(H(:,j)==1);  %第j条边对应的索引
    Dj = D(ind,ind);
    N = length(ind);
    P = zeros(N+1,N+1);
    P(1,:) = [0,ones(1,N)];
    P(:,1) = [0;ones(N,1)];
    P(2:end,2:end) = Dj;   %伪亲和矩阵
    wol(j) = sqrt(abs(det(P)))/(2^(N/2)*factorial(N));
end
miu = mean(wol);
We = exp(-wol/miu);

end


function [L ,AA]= construct_Hypergraphs_knn(W,k_nn)
%计算超图拉普拉斯和超图连接矩阵
% k_nn = 5;
L_H = [];
n_size = size(W,1);
n_vertex = n_size;
n_edge = n_size;
H = zeros(n_edge,n_vertex);
%build Association matrix of Hypergraphs
for i=1:n_vertex
    ll = W(i,:);
    [B,index_i] = sort(ll,'descend');
    k_ii = index_i(1:k_nn);
    H(i,k_ii) = 1;
end

We = cal_hyperedge_weight(W,H');
We = diag(We);
%%%%%
AA = full(Clique_A(We,H));
Dn = diag(sum(AA));
Dn12 = Dn^(-1/2);
Dn12(isnan(Dn12)) = 0;
L = Dn12*(Dn - AA)*Dn12;
AA = Dn12*AA*Dn12;
end


function C = Clique_A(W,H)
%W 表示边的权重
%H 行表示边 列表示点
%N 表示边的个数
N = size(H,1);  %边的个数
A=eye(N);%%A为对角线是1的对角矩阵
if (size(W,2)==1)
    W=diag(W);
end
%X=1/3*H'*W*H;
X=H'*W*(inv(2*A))*H;%%X=HW1/2H因为是3均匀图，所以De是3减去单位矩阵变为2
C = X-diag(diag(X));
C = sparse(C);
end








