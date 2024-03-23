base case:
sigma (i = 0, 0, i) = (0 (0 + 1)) / 2  
 sigma (i = 0, 0, i) 
 by definition of singma 
= 0 
 by arithmetic 
= (0 (0 + 1)) / 2

inductive case:
show sigma (i = 0, n + 1, i) = (n + 1 (n + 1 + 1)) / 2
given: sigma (i = 0, n, i) = (n (n + 1)) / 2
  sigma (i = 0, n + 1, i) 
  by definition 
= sigma (i = 0, n, i) + (n + 1)
  by inductive hypothesis 
= (n (n + 1)) / 2 + (n + 1)
  by arithmetic 
= (n (n + 1)) + (2 (n + 1))）/ 2
= (n + 1 (n + 1 + 1)) / 2

base
sumTo 0 = 0 
inductive case:
 show  sumTo(n + 1) = Sigma (i = 0, n + 1, i)
 given sumTo n = Sigma (i = 0,n, i)

 sumTo (n + 1)
 by def of sumTo, where x is n + 1 
= n + 1 + sumTo (n + 1 - 1)
 by inductive hypothesis 
= (n + 1) + sigma(i = 0, n, i)
 by arithmetic
= sigma(i = 0, n, i) + (n + 1)
 by def of sigma 
= sigma(i = 0, n + 1, i)

