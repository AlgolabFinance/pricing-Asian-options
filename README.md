# pricing-Asian-options
This project try to price an arithmetic Asian option using Binomial tree method and have a comparison with the price of an European options using Black Schole method  


Asian options is well-known as exotic options that have more flexible features compare to plain vanilla options like European options.
There are several pricing methodologies to price Asian options. This report aims to present the Asian option pricing methodology in discrete time, the binomial tree model. 
Using BTM and modified BTM, we will show the numerical results and some sensitivity analysis. While the computational cost of standard BTM is very high, leading to low efficiency, the modified BTM need to be careful with number of representative average prices in order to deduce the correct option values.
We also compare the pros and cons of Asian options over plain vanilla options to persuade both exchanges and investors to enter this market. Lastly, we will discuss the mitigation of the risk of price jump of cryto-currencies.

Step 1: Build the binomial lattice (tree)
We begin by considering an asset whose price at time zero is S_0. In a risk-neutral world, the portfolio is riskless and, for there to be no arbitrage opportunities, it must earn the risk-free interest rate denoted as r. Suppose that the option lasts for maturity T, the length of the time step is t, where t=T/n . In other words, the binomial tree valuation approach involves dividing the life of the option into a large number of small time intervals of length t. It assumes that in each time interval, the price of the underlying asset can either move up from its initial value of S_0 to a new level, S_0u, where, in general, u ≥1, or down from S_0 to a new level, S_0d, where 0<d ≤ 1. 
	The movement from S to S_u , thus, is an “up” movement and the movement from S to S_d is a “down” movement.
	The probability of an up movement will be denote by p. The probability of a down movement is 1 – p.
The notation for the value of the option is shown on the tree. 
![image](https://user-images.githubusercontent.com/85863661/121849516-b6221580-cceb-11eb-9f4e-493af3286bcd.png)
 
	As limited by the scope of this report, we won’t show the proof of BTM calculation. 
	The BTM calculation is shown as follow:

u=1/d  ; u=e^(σ√δt) ; d=e^(-σ√δt)
p=(e^rt-d)/(u-d)
Step 2: Calculate option value at each final node
A_n= {█(1/n ∑_(i=1)^n▒〖S_i,〗  Asian arithmetic@∏_(i=1)^n▒S_i ^(1/n),Asian geometric)┤
Λ(〖S_N,A〗_N)= {█(〖〖(A〗_N-K)〗^+  Asian call,fixed strike@ 〖〖(K-A〗_N)〗^+  Asian put,fixed strike@〖〖(S〗_N-A_N)〗^+  Asian call,floating strike@〖〖(A〗_N-S_N)〗^+  Asian put,floating strike)┤

Step 3: Calculate option value at earlier nodes (backward induction)
a probability p of an up movement, from the (i, j) node at time it to the (i + 1, j + 1) node at time (1 + i) t
a probability 1 - p of a down movement, from the (i, j) node at time it to the (i + 1, j ) node at time (1 + i) t
Λ_t=e^(-rt) [p.Λ_(t+1)^u+(1-p).Λ_(t+1)^d ]
As Asian option is path dependent, for arithmetic type of average, the number of paths to reach certain note in the tree will increase exponentially. The number of representative average for every node at step n is 2^n. This make standard BTM not possible with the number of steps chosen is too high.

	Modified BTM - the Hull White intepolation solution:
The modified BTM is introduced to solve the problem of number of path for arithmetic Asian option. It considers 2 extreme paths which have the following averages:
A_max (N,J)=S_0 (∑_(i=0)^(N-J)▒〖u^i+∑_(i=1)^J▒〖u^(N-J) d^i)/(N+1)〗〗 
A_min (N,J)=S_0 (∑_(i=0)^J▒〖d^i+∑_(i=1)^(N-J)▒〖u^i d^J)/(N+1)〗〗 
At maturity T, the payoff for each path is:
Λ(〖S_N,A〗_max)= {█(〖〖(A〗_max-K)〗^+  Asian call,fixed strike@ 〖〖(K-A〗_max)〗^+  Asian put,fixed strike@〖〖(S〗_N-A_max)〗^+  Asian call,floating strike@〖〖(A〗_max-S_N)〗^+  Asian put,floating strike)┤

Λ(〖S_N,A〗_min)= {█(〖〖(A〗_min-K)〗^+  Asian call,fixed strike@ 〖〖(K-A〗_min)〗^+  Asian put,fixed strike@〖〖(S〗_N-A_min)〗^+  Asian call,floating strike@〖〖(A〗_min-S_N)〗^+  Asian put,floating strike)┤

As the payoff is non-linear, the modified BTM impose equally spaced range from maximum and minimum arithmetic average prices for every node of the tree.
A(i,j,k)=(M-k)/M A_max (i,j)+k/M A_min (i,j)  for k=0,…M

Backward induction for modified BTM:
For A(i,j,k),n≥i≥j≥0 and k= 0…M,
A_u=((i+1)A(i,j,k)+S_0 u^(i+1-j) d^j)/(i+2)
C_u=w_u C(i+1,j,k_u )+(1-w_u)C(i+1,j,k_u-1)
where w_u=(A(i+1,j,k_u-1)-A_u)/(A(i+1,j,k_u-1)-A(i+1,j,k_u ) )

A_d=((i+1)A(i,j,k)+S_0 u^(i-j) d^(j+1))/(i+2)
C_d=w_d C(i+1,j,k_d )+(1-w_d)C(i+1,j,k_d-1)
where w_d=(A(i+1,j,k_d-1)-A_d)/(A(i+1,j,k_d-1)-A(i+1,j,k_d ) )

Choosing the number of representative average prices for each node, M, is also critical to the modified BTM. The pricing results might not converge to exact option values unless M is sufficiently large and well collocated with the number of time steps, n, in the tree model. The more number of time steps in the tree model, the more representative average prices are needed for each node to derive convergent results.


**Data description**

![image](https://user-images.githubusercontent.com/85863661/121849647-e23d9680-cceb-11eb-813a-7392016b56dd.png)



