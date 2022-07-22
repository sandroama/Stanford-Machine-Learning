options = optimset('GradObj','on','MaxIter','100');
initialTheta = zeros(2,1)
[optTheta, functionval, exitFlag] = fminunc(@costFunction, initialTheta, options)


help fminunc // Documentation, function minimum unconstrained