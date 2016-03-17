# include <stdio.h>
# include <stdlib.h>
# include <math.h>
# include "legendre_polynomial.h"

double kernel_cyl(int n, double args[n]){
	int l;
	double r, x, rho, z, th, ang, rb, rk, sigma;
	double r2, z2, x2;
	double *ang_vec;
	double costh[1];

	rho = args[0];
	z = args[1];
	x = args[2];
	rk = args[3];
	sigma = args[4];
	l = args[5];
	 
	r2 = rho * rho;
	z2 = z * z;
	x2 = x * x;

	r = sqrt(r2 +  z2);
	th = atan2(rho, z);
	costh[0] = cos(th);
	ang_vec = p_polynomial_value(1, l, costh);
	ang = ang_vec[l];
	free ( ang_vec );

	sigma *= sigma;
	rb = (r - rk);
	rb = rb * rb / sigma;
	rb = exp(-rb);
	rb *= ang;

	//printf("%g ", ang);

	return rb * rho / sqrt(r2 - x2);
};

double kernel_pol(int n, double args[n]){
	int l;
	double r, x, th, ang, rb, rk, sigma;
	double r2, z2, x2;
	double *ang_vec;
	double costh[1];

	r = args[0];
	th = args[1];
	x = args[2];
	rk = args[3];
	sigma = args[4];
	l = args[5];
	 
	r2 = r * r;
	x2 = x * x;

	costh[0] = cos(th);
	ang_vec = p_polynomial_value(1, l, costh);
	ang = ang_vec[l];
	free ( ang_vec );

	sigma *= sigma;
	rb = (r - rk);
	rb = rb * rb / sigma;
	rb = exp(-rb);
	rb *= ang;

	//printf("%g ", ang);

	return rb * r / sqrt(r2 - x2);
};
